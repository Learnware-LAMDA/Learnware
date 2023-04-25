import numpy as np
import torch
from typing import Union


class BaseModel:
    """Base interface tor model standard when user want to submit learnware to market."""

    def __init__(self, input_shape: tuple, output_shape: tuple):
        """The initialization method for base model

        Parameters
        ----------
        input_shape : tuple
            The shape of input features, which must be given when inherit BaseModel, could be used for checking learnware
        output_shape : tuple
            The shape of output prediction, which must be given when inherit BaseModel, could be used for checking learnware
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def predict(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        """The prediction method for model in learnware, which will be checked when learnware is submitted into the market.

        Parameters
        ----------
        X : Union[np.ndarray, torch.tensor]
            The features array for prediciton
        Returns
        -------
        Union[np.ndarray, torch.tensor]
            The predictions array
        """
        pass

    def fit(self, X: Union[np.ndarray, torch.tensor], y: Union[np.ndarray, torch.tensor]):
        pass

    def finetune(self, X: Union[np.ndarray, torch.tensor], y: np.ndarray):
        """The finetune method for continuing train the model searched by market

        Parameters
        ----------
        X : Union[np.ndarray, torch.tensor]
            features for finetuning
        y : np.ndarray
            labels for finetuning
        """
        pass
