import numpy as np
from typing import List

from ..evolve.organizer import EvolvedOrganizer
from ...learnware import Learnware


class MappingFunction:
    def __init__(self) -> None:
        pass

    def transform(X: np.ndarray) -> np.ndarray:
        """transform the data in one feature space to another feature space.

        Parameters
        ----------
        X : np.ndarray
            data in one feature space

        Returns
        -------
        np.ndarray
            transformed data in other feature space
        """
        pass


class HeterogeneousOrganizer(EvolvedOrganizer):
    """Organize learnwares with heterogeneous feature spaces, organizer version with evolved learnwares"""

    def __init__(self, *args, **kwargs):
        super(HeterogeneousOrganizer, self).__init__(*args, **kwargs)
        self.mapping_function_list = {}

    def _mapping_function_list_initialization(self, learnware_list: List[Learnware]):
        """Initialize mapping functions with all submitted learnwares

        Parameters
        ----------
        learnware_list : List[Learnware]
            list of learnwares
        """
        self.mapping_function_list = self.learn_mapping_functions(learnware_list)

    def learn_mapping_functions(self, learnware_list: List[Learnware]) -> List[MappingFunction]:
        """Use all statistical specifications of submitted learnwares to generate mapping functions from each original feature space to subsapce and vice verse.

        Parameters
        ----------
        learnware_list : List[Learnware]
            list of learnwares

        Returns
        -------
        List[MappingFunction]
            list of mapping functions
        """
        pass

    def transform_original_to_subspace(
        self, original_feature_space_idx: int, original_feature: np.ndarray
    ) -> np.ndarray:
        """Transform feature in a original feature space to the subspace.

        Parameters
        ----------
        original_feature_space_idx : int
            index of the original feature space
        original_feature : np.ndarray
            data in the original feature space

        Returns
        -------
        np.ndarray
            mapped data in the subspace
        """
        pass

    def transform_subspace_to_original(
        self, original_feature_space_idx: int, subspace_feature: np.ndarray
    ) -> np.ndarray:
        """Transform feature in the subspace to a original feature space.

        Parameters
        ----------
        original_feature_space_idx : int
            index of the original feature space
        subspace_feature : np.ndarray
            data in the subspace

        Returns
        -------
        np.ndarray
            mapped data in the original feature space
        """
        pass
