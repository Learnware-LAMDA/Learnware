import os
import pickle

import numpy as np

from learnware.model import BaseModel


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__(input_shape=(1,), output_shape=(1,))
        dir_path = os.path.dirname(os.path.abspath(__file__))

        modelv_path = os.path.join(dir_path, "modelv.pth")
        with open(modelv_path, "rb") as f:
            self.modelv = pickle.load(f)

        modell_path = os.path.join(dir_path, "modell.pth")
        with open(modell_path, "rb") as f:
            self.modell = pickle.load(f)

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.modell.predict(self.modelv.transform(X))

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
