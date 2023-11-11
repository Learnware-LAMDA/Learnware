import numpy as np

from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from .base import BaseReuser
from ..learnware import Learnware


class FeatureAugmentReuser(BaseReuser):
    """
    FeatureAugmentReuser is a class for augmenting features using predictions of a given learnware model and applying regression or classification on the augmented dataset.

    This class supports two modes:
    - "regression": Uses RidgeCV for regression tasks.
    - "classification": Uses LogisticRegressionCV for classification tasks.
    """

    def __init__(self, learnware: Learnware = None, mode: str = None):
        """
        Initialize the FeatureAugmentReuser with a learnware model and a mode.

        Parameters
        ----------
        learnware : Learnware
            A learnware model used for initial predictions.
        mode : str
            The mode of operation, either "regression" or "classification".
        """
        self.learnware = learnware
        assert mode in ["classification", "regression"], "Mode must be either 'classification' or 'regression'"
        self.mode = mode

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
        user_data = self._fill_data(user_data)
        y_pred = self.learnware.predict(user_data)
        user_data_aug = np.concatenate((user_data, y_pred.reshape(-1, 1)), axis=1)
        y_pred_aug = self.output_aligner.predict(user_data_aug)
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
        x_train = self._fill_data(x_train)
        y_pred = self.learnware.predict(x_train)
        x_train_aug = np.concatenate((x_train, y_pred.reshape(-1, 1)), axis=1)
        if self.mode == "regression":
            alpha_list = [0.01, 0.1, 1.0, 10, 100]
            ridge_cv = RidgeCV(alphas=alpha_list, store_cv_values=True)
            ridge_cv.fit(x_train_aug, y_train)
            self.output_aligner = ridge_cv
        elif self.mode == "classification":
            self.output_aligner = LogisticRegressionCV(cv=5, max_iter=1000, random_state=0, multi_class="auto")
            self.output_aligner.fit(x_train_aug, y_train)

    def _fill_data(self, X: np.ndarray):
        """
        Fill missing data (NaN, Inf) in the input array with the mean of the column.

        Parameters
        ----------
        X : np.ndarray
            Input data array that may contain missing values.

        Returns
        -------
        np.ndarray
            Data array with missing values filled.

        Raises
        ------
        ValueError
            If a column in X contains only exceptional values (NaN, Inf).
        """
        X[np.isinf(X) | np.isneginf(X) | np.isposinf(X) | np.isneginf(X)] = np.nan
        if np.any(np.isnan(X)):
            for col in range(X.shape[1]):
                is_nan = np.isnan(X[:, col])
                if np.any(is_nan):
                    if np.all(is_nan):
                        raise ValueError(f"All values in column {col} are exceptional, e.g., NaN and Inf.")
                    col_mean = np.nanmean(X[:, col])
                    X[:, col] = np.where(is_nan, col_mean, X[:, col])
        return X
