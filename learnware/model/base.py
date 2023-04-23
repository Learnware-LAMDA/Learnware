import numpy as np
from abc import abstractmethod


class BaseModel:
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
