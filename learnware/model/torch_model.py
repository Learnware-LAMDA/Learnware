import numpy as np
from torch import nn


class TorchModel:
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple,
        output_shape: tuple,
    ):
        self._model = model
        self.input_shape = input_shape
        self.output_shape = output_shape

    @property
    def nn_model(self) -> nn.Module:
        """
        fetch the inner model
        """
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
