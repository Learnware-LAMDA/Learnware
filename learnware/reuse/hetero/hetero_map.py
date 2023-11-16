import numpy as np

from ..align import AlignLearnware
from ...learnware import Learnware
from ...logger import get_module_logger
from .feature_align import FeatureAlignLearnware
from ..feature_augment import FeatureAugmentReuser
from ...specification import RKMETableSpecification

logger = get_module_logger("hetero_map_align")


class HeteroMapAlignLearnware(AlignLearnware):
    """
    HeteroMapAlignLearnware is a class designed for reusing learnware models with feature alignment and augmentation.
    It can handle both classification and regression tasks and supports fine-tuning on additional training data.

    Attributes
    ----------
    learnware : Learnware
        The learnware model to be reused.
    mode : str
        The mode of operation, either "classification" or "regression".
    cuda_idx : int
        Index of the CUDA device to be used for computations.
    align_arguments : dict
        Additional arguments for feature alignment.
    """

    def __init__(self, learnware: Learnware = None, mode: str = None, cuda_idx=0, **align_arguments):
        """
        Initialize the HeteroMapAlignLearnware with a learnware model, mode, CUDA device index, and alignment arguments.

        Parameters
        ----------
        learnware : Learnware
            A learnware model used for initial predictions.
        mode : str
            The mode of operation, either "regression" or "classification".
        cuda_idx : int
            The index of the CUDA device for computations.
        align_arguments : dict
            Additional arguments to be passed to the feature alignment process.
        """
        super(HeteroMapAlignLearnware, self).__init__(learnware)
        assert mode in ["classification", "regression"], "Mode must be either 'classification' or 'regression'"
        self.mode = mode
        self.cuda_idx = cuda_idx
        self.align_arguments = align_arguments
        self.reuser = None

    def align(self, user_rkme: RKMETableSpecification, x_train: np.ndarray = None, y_train: np.ndarray = None):
        """
        Align the hetero learnware using the user RKME specification and labeled data.

        Parameters
        ----------
        user_rkme : RKMETableSpecification
            The RKME specification from the user dataset.
        x_train : ndarray
            Training data features.
        y_train : ndarray
            Training data labels.
        """
        self.feature_align_learnware = FeatureAlignLearnware(
            learnware=self, cuda_idx=self.cuda_idx, **self.align_arguments
        )
        self.feature_align_learnware.align(user_rkme)

        if x_train is None or y_train is None:
            logger.warning("Hetero learnware may not perform well as labeled data alignment is not provided!")
            self.reuser = self.feature_align_learnware
        else:
            self.reuser = FeatureAugmentReuser(learnware_list=[self.feature_align_learnware], mode=self.mode)
            self.reuser.fit(x_train, y_train)

    def predict(self, user_data):
        """
        Predict the output for user data using the feature aligner or the fine-tuned model.

        Parameters
        ----------
        user_data : ndarray
            Input data for making predictions.

        Returns
        -------
        ndarray
            Predicted output from the model.
        """
        assert self.reuser is not None, "HeteroMapAlignLearnware must be aligned before making predictions."
        return self.reuser.predict(user_data)
