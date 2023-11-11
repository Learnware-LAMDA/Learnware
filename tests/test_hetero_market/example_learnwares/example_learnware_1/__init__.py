from learnware.model import BaseModel
import numpy as np
import joblib
import os


class MyModel(BaseModel):
    def __init__(self):
        super(MyModel, self).__init__(input_shape=(30,), output_shape=(1,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(dir_path, "ridge.pkl")
        model = joblib.load(model_path)
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
