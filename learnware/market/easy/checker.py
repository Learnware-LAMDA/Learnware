import traceback

from ..base import LearnwareChecker
from ...logger import get_module_logger

logger = get_module_logger("easy_checker", "INFO")

class EasyChecker(LearnwareChecker):
        
    def __call__(self, learnware):
        semantic_spec = learnware.get_specification().get_semantic_spec()

        try:
            # check model instantiation
            learnware.instantiate_model()

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"The learnware [{learnware.id}] is instantiated failed! Due to {e}")
            return self.NONUSABLE_LEARNWARE

        try:
            learnware_model = learnware.get_model()

            # check input shape
            if semantic_spec["Data"]["Values"][0] == "Table":
                input_shape = (semantic_spec["Input"]["Dimension"],)
            else:
                input_shape = learnware_model.input_shape
                pass

            # check rkme dimension
            stat_spec = learnware.get_specification().get_stat_spec_by_name("RKMEStatSpecification")
            if stat_spec is not None:
                if stat_spec.get_z().shape[1:] != input_shape:
                    logger.warning(f"The learnware [{learnware.id}] input dimension mismatch with stat specification")
                    return self.NONUSABLE_LEARNWARE
                pass

            inputs = np.random.randn(10, *input_shape)
            outputs = learnware.predict(inputs)

            # check output
            if outputs.ndim == 1:
                outputs = outputs.reshape(-1, 1)
                pass

            if semantic_spec["Task"]["Values"][0] in ("Classification", "Regression", "Feature Extraction"):
                # check output type
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.detach().cpu().numpy()
                if not isinstance(outputs, np.ndarray):
                    logger.warning(f"The learnware [{learnware.id}] output must be np.ndarray or torch.Tensor")
                    return self.NONUSABLE_LEARNWARE

                # check output shape
                output_dim = int(semantic_spec["Output"]["Dimension"])
                if outputs[0].shape[0] != output_dim:
                    logger.warning(f"The learnware [{learnware.id}] input and output dimention is error")
                    return self.NONUSABLE_LEARNWARE
                pass
            else:
                if outputs.shape[1:] != learnware_model.output_shape:
                    logger.warning(f"The learnware [{learnware.id}] input and output dimention is error")
                    return self.NONUSABLE_LEARNWARE

        except Exception as e:
            logger.warning(f"The learnware [{learnware.id}] prediction is not avaliable! Due to {repr(e)}")
            return self.NONUSABLE_LEARNWARE

        return self.USABLE_LEARWARE
