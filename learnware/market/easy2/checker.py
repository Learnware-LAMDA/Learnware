import traceback
import numpy as np
import torch
import random
import string

from ..base import BaseChecker
from ..utils import parse_specification_type
from ...config import C
from ...logger import get_module_logger

logger = get_module_logger("easy_checker", "INFO")


class EasySemanticChecker(BaseChecker):
    @staticmethod
    def check_semantic_spec(semantic_spec):
        try:
            for key in C["semantic_specs"]:
                value = semantic_spec[key]["Values"]
                valid_type = C["semantic_specs"][key]["Type"]
                assert semantic_spec[key]["Type"] == valid_type, f"{key} type mismatch"

                if valid_type == "Class":
                    valid_list = C["semantic_specs"][key]["Values"]
                    assert len(value) == 1, f"{key} must be unique"
                    assert value[0] in valid_list, f"{key} must be in {valid_list}"

                elif valid_type == "Tag":
                    valid_list = C["semantic_specs"][key]["Values"]
                    assert len(value) >= 1, f"{key} cannot be empty"
                    for v in value:
                        assert v in valid_list, f"{key} must be in {valid_list}"

                elif valid_type == "String":
                    assert isinstance(value, str), f"{key} must be string"
                    assert len(value) >= 1, f"{key} cannot be empty"

            if semantic_spec["Data"]["Values"][0] == "Table":
                assert semantic_spec["Input"] is not None, "Lack of input semantics"
                dim = semantic_spec["Input"]["Dimension"]
                for k, v in semantic_spec["Input"]["Description"].items():
                    assert int(k) >= 0 and int(k) < dim, f"Dimension number in [0, {dim})"
                    assert isinstance(v, str), "Description must be string"

            if semantic_spec["Task"]["Values"][0] in ["Classification", "Regression", "Feature Extraction"]:
                assert semantic_spec["Output"] is not None, "Lack of output semantics"
                dim = semantic_spec["Output"]["Dimension"]
                for k, v in semantic_spec["Output"]["Description"].items():
                    assert int(k) >= 0 and int(k) < dim, f"Dimension number in [0, {dim})"
                    assert isinstance(v, str), "Description must be string"

            return EasySemanticChecker.NONUSABLE_LEARNWARE

        except AssertionError as err:
            logger.warning(f"semantic_specification is not valid due to {err}!")
            return EasySemanticChecker.INVALID_LEARNWARE

    def __call__(self, learnware):
        semantic_spec = learnware.get_specification().get_semantic_spec()
        return self.check_semantic_spec(semantic_spec)


class EasyStatChecker(BaseChecker):
    @staticmethod
    def _generate_random_text_list(num, text_type="en", min_len=10, max_len=1000):
        text_list = []
        for i in range(num):
            length = random.randint(min_len, max_len)
            if text_type == "en":
                characters = string.ascii_letters + string.digits + string.punctuation
                result_str = "".join(random.choice(characters) for i in range(length))
                text_list.append(result_str)
            elif text_type == "zh":
                result_str = "".join(chr(random.randint(0x4E00, 0x9FFF)) for i in range(length))
                text_list.append(result_str)
            else:
                raise ValueError("Type should be en or zh")
        return text_list

    def __call__(self, learnware):
        semantic_spec = learnware.get_specification().get_semantic_spec()

        try:
            # Check model instantiation
            learnware.instantiate_model()

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"The learnware [{learnware.id}] is instantiated failed! Due to {e}.")
            return self.INVALID_LEARNWARE

        try:
            learnware_model = learnware.get_model()
            # Check input shape
            input_shape = learnware_model.input_shape

            if semantic_spec["Data"]["Values"][0] == "Table" and input_shape != (
                int(semantic_spec["Input"]["Dimension"]),
            ):
                logger.warning("input shapes of model and semantic specifications are different")
                return self.INVALID_LEARNWARE

            spec_type = parse_specification_type(learnware.get_specification().stat_spec)
            if spec_type is None:
                logger.warning(f"No valid specification is found in stat spec {spec_type}")
                return self.INVALID_LEARNWARE

            if spec_type == "RKMETableSpecification":
                stat_spec = learnware.get_specification().get_stat_spec_by_name(spec_type)
                if stat_spec.get_z().shape[1:] != input_shape:
                    logger.warning(f"The learnware [{learnware.id}] input dimension mismatch with stat specification.")
                    return self.INVALID_LEARNWARE
                inputs = np.random.randn(10, *input_shape)
            elif spec_type == "RKMETextSpecification":
                inputs = EasyStatChecker._generate_random_text_list(10)
            elif spec_type == "RKMEImageSpecification":
                inputs = np.random.randint(0, 255, size=(10, *input_shape))
            else:
                raise ValueError(f"not supported spec type for spec_type = {spec_type}")

            # Check output
            try:
                outputs = learnware.predict(inputs)
            except Exception:
                logger.warning(f"learnware {learnware} prediction method is not valid!")
                return self.INVALID_LEARNWARE

            if semantic_spec["Task"]["Values"][0] in ("Classification", "Regression", "Feature Extraction"):
                # Check output type
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.detach().cpu().numpy()
                if not isinstance(outputs, np.ndarray):
                    logger.warning(f"The learnware [{learnware.id}] output must be np.ndarray or torch.Tensor!")
                    return self.INVALID_LEARNWARE

                if outputs.ndim == 1:
                    outputs = outputs.reshape(-1, 1)
                # Check output shape
                if outputs[0].shape != learnware_model.output_shape or learnware_model.output_shape != (
                    int(semantic_spec["Output"]["Dimension"]),
                ):
                    logger.warning(
                        f"The learnware [{learnware.id}] output dimension mismatch!, where pred_shape={outputs[0].shape}, model_shape={learnware_model.output_shape}, semantic_shape={(int(semantic_spec['Output']['Dimension']), )}"
                    )
                    return self.INVALID_LEARNWARE

        except Exception as e:
            logger.warning(f"The learnware [{learnware.id}] prediction is not avaliable! Due to {repr(e)}.")
            return self.INVALID_LEARNWARE

        return self.USABLE_LEARWARE
