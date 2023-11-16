import torch
import numpy as np
from typing import List, Union
from scipy.special import softmax


from ..learnware import Learnware
from .base import BaseReuser
from ..logger import get_module_logger

logger = get_module_logger("avaraging_reuser")


class AveragingReuser(BaseReuser):
    """Baseline Multiple Learnware Reuser using Ensemble Method"""

    def __init__(self, learnware_list: List[Learnware] = None, mode: str = "mean"):
        """The initialization method for averaging ensemble reuser

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list contains learnwares.
        mode : str
            - "mean": average the output of all learnwares for regression task (learnware output is a real number)
            - "vote_by_label": vote by labels for classification task, learnware output belongs to the set {0, 1, ..., class_num}
            - "vote_by_prob": vote by probabilities for classification task, learnware output is a logits vector, denoting the probability of each class
        """
        super(AveragingReuser, self).__init__(learnware_list)
        if mode not in ["mean", "vote_by_label", "vote_by_prob"]:
            raise ValueError(f"Mode must be one of ['mean', 'vote_by_label', 'vote_by_prob'], but got {mode}")
        self.mode = mode

    def predict(self, user_data: np.ndarray) -> np.ndarray:
        """Prediction for user data using baseline ensemble method

        Parameters
        ----------
        user_data : np.ndarray
            Raw user data.

        Returns
        -------
        np.ndarray
            Prediction given by ensemble method
        """
        preds = []
        for learnware in self.learnware_list:
            pred_y = learnware.predict(user_data)
            if isinstance(pred_y, torch.Tensor):
                pred_y = pred_y.detach().cpu().numpy()
            if not isinstance(pred_y, np.ndarray):
                raise TypeError(f"Model output must be np.ndarray or torch.Tensor")

            if len(pred_y.shape) == 1:
                pred_y = pred_y.reshape(-1, 1)
            else:
                if self.mode == "vote_by_label":
                    if pred_y.shape[1] > 1:
                        pred_y = pred_y.argmax(axis=1).reshape(-1, 1)
                elif self.mode == "vote_by_prob":
                    pred_y = softmax(pred_y, axis=-1)
            preds.append(pred_y)

        if self.mode == "vote_by_prob":
            return np.mean(preds, axis=0)
        else:
            preds = np.concatenate(preds, axis=1)
            if self.mode == "mean":
                return preds.mean(axis=1)
            elif self.mode == "vote_by_label":
                return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)
