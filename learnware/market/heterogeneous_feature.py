import numpy as np
from typing import Tuple, Any, List, Union, Dict

from .evolve import EvolvedMarket
from ..learnware import Learnware


class HeterogeneousFeatureMarket(EvolvedMarket):
    """Organize learnwares with heterogeneous feature spaces

    Parameters
    ----------
    EvolvedMarket : _type_
        Market version with evolved learnwares
    """

    def __init__(self):
        super(EvolvedMarket, self).__init__()

    def learn_mapping_functions(self, learnware_list: List[Learnware]):
        """Use all statistical specifications of submitted learnwares to generate mapping functions from each original feature space to subsapce and vice verse.

        Parameters
        ----------
        learnware_list : List[Learnware]
            list of learnwares
        """
        pass

    def transform_original_to_subspace(self, original_feature_space_idx: int, original_feature: np.ndarray):
        """Transform feature in a original feature space to the subspace.

        Parameters
        ----------
        original_feature_space_idx: int
            index of the original feature space
        original_feature: np.ndarray
            data in the original feature space
        """
        pass

    def transform_subspace_to_original(self, original_feature_space_idx: int, subspace_feature: np.ndarray):
        """Transform feature in a original feature space to the subspace.

        Parameters
        ----------
        original_feature_space_idx: int
            index of the original feature space
        subspace_feature: np.ndarray
            data in the subspace
        """
        pass
