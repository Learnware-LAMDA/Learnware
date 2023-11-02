import numpy as np
from typing import List

from ..learnware import Learnware
from ..logger import get_module_logger

logger = get_module_logger("reuser")


class BaseReuser:
    """Providing the interfaces to reuse the learnwares which is searched by learnware"""

    def __init__(self, learnware_list: List[Learnware] = None):
        """The initializaiton method for base reuser

        Parameters
        ----------
        learnware_list : List[Learnware]
            The learnware list to reuse and make predictions
        """
        self.learnware_list = learnware_list

    def reset(self, **kwargs):
        for _k, _v in kwargs.items():
            if hasattr(_k):
                setattr(_k, _v)

    def predict(self, user_data: np.ndarray) -> np.ndarray:
        """Give the final prediction for user data with reused learnware

        Parameters
        ----------
        user_data : np.ndarray
            User's unlabeled raw data.

        Returns
        -------
        np.ndarray
            The final prediction for user data with reused learnware
        """
        raise NotImplementedError("The predict method is not implemented!")
