import os
import joblib
import numpy as np


class SVM:
    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(os.path.join(dir_path, 'svm.pkl'))

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def fintune(self, X: np.ndarray, y: np.ndarray):
        pass