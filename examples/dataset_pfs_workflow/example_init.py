import os
import joblib
import numpy as np
from learnware.model import BaseModel


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__(input_shape=(31,), output_shape=(1,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(os.path.join(dir_path, "model.out"))

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
