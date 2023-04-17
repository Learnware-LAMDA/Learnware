import numpy as np
from abc import abstractmethod


class BaseModel:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
