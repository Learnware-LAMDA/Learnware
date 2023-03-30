import numpy as np

class BaseModel:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fintune(self, X: np.ndarray, y: np.ndarray):
        pass
