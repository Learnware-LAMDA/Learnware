import os
import pickle
import numpy as np
from learnware.model import BaseModel


class SVM(BaseModel):
    def __init__(self):
        super(SVM, self).__init__(input_shape=(64,), output_shape=(10,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.model = pickle.load(os.path.join(dir_path, "svm.pkl"))

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass
