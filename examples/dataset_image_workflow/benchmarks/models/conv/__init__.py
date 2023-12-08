import os

import torch
import numpy as np
from learnware.model import BaseModel

from .model import ConvModel


class Model(BaseModel):
    def __init__(self, device="cuda", input_channel=3):
        super(Model, self).__init__(input_shape=(input_channel, 32, 32), output_shape=(10,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.device =device
        self.model = ConvModel(channel=input_channel, n_random_features=10)
        self.model.load_state_dict(torch.load(os.path.join(dir_path, "model.pth")))
        self.model.to(device).eval()

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model(torch.asarray(X, dtype=torch.float32, device=self.device))

    def __call__(self, *args, **kwargs):
        self.predict(*args, **kwargs)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError()
