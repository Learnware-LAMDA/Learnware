import os
import joblib
import numpy as np
from learnware.model import BaseModel
from model import ConvModel
import torch


class Model(BaseModel):
    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvModel(channel=3, n_random_features=10).to(device)
        self.model.load_state_dict(torch.load(os.path.join(dir_path, "conv_model.pth")))
        self.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
