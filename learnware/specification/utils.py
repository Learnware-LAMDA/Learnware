import torch
import numpy as np
import pandas as pd
from typing import Union


def convert_to_numpy(data: Union[np.ndarray, pd.DataFrame, torch.Tensor]):
    """Convert data to np.ndarray

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame, or torch.Tensor
        The input data that needs to be converted to a NumPy array.

    Returns
    -------
    np.ndarray
        The data converted to a NumPy array.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        raise TypeError(
            "Unsupported data format. Please provide a NumPy array, a Pandas DataFrame, or a PyTorch Tensor."
        )
