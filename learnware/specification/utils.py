import torch
import numpy as np
import pandas as pd
from typing import Union

from .base import BaseStatSpecification
from .rkme import RKMEStatSpecification
from ..config import C


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
        raise TypeError("Unsupported data format. Please provide a NumPy array, a Pandas DataFrame, or a PyTorch Tensor.")


def generate_rkme_spec(
    X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    gamma: float = 0.1,
    reduced_set_size: int = 100,
    step_size: float = 0.1,
    steps: int = 3,
    nonnegative_beta: bool = True,
    reduce: bool = True,
    cuda_idx: int = None,
) -> RKMEStatSpecification:
    """
        Interface for users to generate Reduced Kernel Mean Embedding (RKME) specification.
        Return a RKMEStatSpecification object, use .save() method to save as json file.

    Parameters
    ----------
    X : np.ndarray, pd.DataFrame, or torch.Tensor
        Raw data in np.ndarray, pd.DataFrame, or torch.Tensor format.
        The shape of X:
            First dimension represents the number of samples (data points).
            The remaining dimensions represent the dimensions (features) of each sample.
            For example, if X has shape (100, 3), it means there are 100 samples, and each sample has 3 features.
    gamma : float
        Bandwidth in gaussian kernel, by default 0.1.
    reduced_set_size : int
        Size of the construced reduced set.
    step_size : float
        Step size for gradient descent in the iterative optimization.
    steps : int
        Total rounds in the iterative optimization.
    nonnegative_beta : bool, optional
        True if weights for the reduced set are intended to be kept non-negative, by default False.
    reduce : bool, optional
        Whether shrink original data to a smaller set, by default True
    cuda_idx : int
        A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used.
        None indicates that CUDA is automatically selected.

    Returns
    -------
    RKMEStatSpecification
        A RKMEStatSpecification object
    """
    # Convert data type
    X = convert_to_numpy(X)
    X = np.ascontiguousarray(X).astype(np.float32)
    
    # Check reduced_set_size
    max_reduced_set_size = C.max_reduced_set_size
    if reduced_set_size * X[0].size > max_reduced_set_size:
        reduced_set_size = max(20, max_reduced_set_size // X[0].size)
    
    # Check cuda_idx
    if not torch.cuda.is_available() or cuda_idx == -1:
        cuda_idx = -1
    else:
        num_cuda_devices = torch.cuda.device_count()
        if cuda_idx is None or not (cuda_idx >= 0 and cuda_idx < num_cuda_devices):
            cuda_idx = 0
    
    # Generate rkme spec
    rkme_spec = RKMEStatSpecification(gamma=gamma, cuda_idx=cuda_idx)
    rkme_spec.generate_stat_spec_from_data(X, reduced_set_size, step_size, steps, nonnegative_beta, reduce)
    return rkme_spec


def generate_stat_spec(X: np.ndarray) -> BaseStatSpecification:
    """
        Interface for users to generate statistical specification.
        Return a StatSpecification object, use .save() method to save as npy file.

    Parameters
    ----------
    X : np.ndarray
        Raw data in np.ndarray format.
        Size of array: (n*d)

    Returns
    -------
    StatSpecification
        A StatSpecification object
    """
    return None