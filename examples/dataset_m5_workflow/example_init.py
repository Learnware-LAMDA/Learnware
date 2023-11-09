import os
import joblib
import numpy as np
import lightgbm as lgb
from learnware.model import BaseModel


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__(input_shape=(82,), output_shape=(1,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.model = lgb.Booster(model_file=os.path.join(dir_path, "model.out"))

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
