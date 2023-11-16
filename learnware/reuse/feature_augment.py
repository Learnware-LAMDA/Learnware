import torch
import numpy as np
from typing import List
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

from .base import BaseReuser
from .utils import fill_data_with_mean
from ..learnware import Learnware


class FeatureAugmentReuser(BaseReuser):
    """
    FeatureAugmentReuser is a class for augmenting features using predictions of a given learnware model and applying regression or classification on the augmented dataset.

    This class supports two modes:
        - "regression": Uses RidgeCV for regression tasks.
        - "classification": Uses LogisticRegressionCV for classification tasks.
    """

    def __init__(self, learnware_list: List[Learnware] = None, mode: str = None):
        """
        Initialize the FeatureAugmentReuser with a learnware model and a mode.

        Parameters
        ----------
        learnware : List[Learnware]
            The list contains learnwares.
        mode : str
            The mode of operation, either "regression" or "classification".
        """
        super(FeatureAugmentReuser, self).__init__(learnware_list)
        assert mode in ["classification", "regression"], "Mode must be either 'classification' or 'regression'"
        self.mode = mode
        self.augment_reuser = None

    def predict(self, user_data: np.ndarray) -> np.ndarray:
        """
        Predict the output for user data using the trained output aligner model.

        Parameters
        ----------
        user_data : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            Predicted output from the output aligner model.
        """
        assert self.augment_reuser is not None, "FeatureAugmentReuser is not trained by labeled data yet."

        user_data = fill_data_with_mean(user_data)
        user_data_aug = self._get_augment_data(user_data)
        y_pred_aug = self.augment_reuser.predict(user_data_aug)

        return y_pred_aug

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Train the output aligner model using the training data augmented with predictions from the learnware model.

        Parameters
        ----------
        x_train : np.ndarray
            Training data features.
        y_train : np.ndarray
            Training data labels.
        """
        x_train = fill_data_with_mean(x_train)
        x_train_aug = self._get_augment_data(x_train)

        if self.mode == "regression":
            alpha_list = [0.01, 0.1, 1.0, 10, 100]
            ridge_cv = RidgeCV(alphas=alpha_list, store_cv_values=True)
            ridge_cv.fit(x_train_aug, y_train)
            self.augment_reuser = ridge_cv
        else:
            self.augment_reuser = LogisticRegressionCV(cv=5, max_iter=1000, random_state=0, multi_class="auto")
            self.augment_reuser.fit(x_train_aug, y_train)

    def _get_augment_data(self, X: np.ndarray) -> np.ndarray:
        """Get the augmented data with model output.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Augment data with model output.

        Raises
        ------
        TypeError
            If the type of model output not in [np.ndarray, torch.Tensor].
        """
        y_preds = []
        for learnware in self.learnware_list:
            y_pred = learnware.predict(X)
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
            if not isinstance(y_pred, np.ndarray):
                raise TypeError(f"Model output must be np.ndarray or torch.Tensor")
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
            y_preds.append(y_pred)
        y_preds = np.concatenate(y_preds, axis=1)

        return np.concatenate((X, y_preds), axis=1)
