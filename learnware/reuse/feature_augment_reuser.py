from typing import List
import numpy as np

from sklearn.linear_model import RidgeCV

from .base import BaseReuser
from learnware.learnware import Learnware


class FeatureAugmentReuser(BaseReuser):
    def __init__(self, learnware: Learnware = None, task_type: str = None):
        self.learnware=learnware
        assert task_type in ["classification", "regression"]
        self.task_type=task_type

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        x_test=self._fill_data(x_test)
        y_pred=self.learnware.predict(x_test)
        x_test_aug=np.concatenate((x_test, y_pred.reshape(-1, 1)), axis=1)
        y_pred_aug=self.output_aligner.predict(x_test_aug)
        return y_pred_aug

    def fit(self, x_train, y_train):
        x_train=self._fill_data(x_train)
        y_pred=self.learnware.predict(x_train)
        x_train_aug=np.concatenate((x_train, y_pred.reshape(-1, 1)), axis=1)
        if self.task_type=="regression":
            alpha_list = [0.01, 0.1, 1.0, 10, 100]
            ridge_cv = RidgeCV(alphas=alpha_list, store_cv_values=True)
            ridge_cv.fit(x_train_aug, y_train)
            self.output_aligner=ridge_cv
        elif self.task_type=="classification":
            raise NotImplementedError("Not implemented yet!")

    def _fill_data(self, X: np.ndarray):
        X[np.isinf(X) | np.isneginf(X) | np.isposinf(X) | np.isneginf(X)] = np.nan
        if np.any(np.isnan(X)):
            for col in range(X.shape[1]):
                is_nan = np.isnan(X[:, col])
                if np.any(is_nan):
                    if np.all(is_nan):
                        raise ValueError(f"All values in column {col} are exceptional, e.g., NaN and Inf.")
                    # Fill np.nan with np.nanmean
                    col_mean = np.nanmean(X[:, col])
                    X[:, col] = np.where(is_nan, col_mean, X[:, col])
        return X