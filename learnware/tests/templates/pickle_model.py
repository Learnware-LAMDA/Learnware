import os
import pickle
import numpy as np
from learnware.model.base import BaseModel

class PickleLoadedModel(BaseModel):
    
    def __init__(
        self,
        input_shape,
        output_shape,
        predict_method="predict",
        fit_method="fit",
        finetune_method="finetune",
        pickle_filename="model.pkl",
    ):
        super(PickleLoadedModel, self).__init__(input_shape=input_shape, output_shape=output_shape)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.pickle_filepath = os.path.join(dir_path, pickle_filename)
        with open(self.pickle_filepath, "rb") as fd:
            self.model = pickle.load(fd)
        self.predict_method = predict_method
        self.fit_method = fit_method
        self.finetune_method = finetune_method
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        return getattr(self.model, self.predict_method)(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        getattr(self.model, self.fit_method)(X, y)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        getattr(self.model, self.finetune_method)(X, y)
