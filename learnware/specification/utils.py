import numpy as np

from .base import BaseStatSpecification
from .rkme import RKMEStatSpecification


def generate_rkme_spec(
    X: np.ndarray,
    gamma: float = 0.1,
    K: int = 100,
    step_size: float = 0.1,
    steps: int = 3,
    nonnegative_beta: bool = True,
    reduce: bool = True,
    cuda_idx: int = -1,
) -> RKMEStatSpecification:
    """
            Interface for users to generate Reduced-set Kernel Mean Embedding (RKME) specification.
            Return a RKMEStatSpecification object, use .save() method to save as json file.


    Parameters
    ----------
    X : np.ndarray
            Raw data in np.ndarray format.
            Size of array: (n*d)
    gamma : float
    Bandwidth in gaussian kernel, by default 0.1.
    K : int
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

    Returns
    -------
    RKMEStatSpecification
            A RKMEStatSpecification object
    """
    X = np.ascontiguousarray(X).astype(np.float32)
    rkme_spec = RKMEStatSpecification(gamma=gamma, cuda_idx=cuda_idx)
    rkme_spec.generate_stat_spec_from_data(X, K, step_size, steps, nonnegative_beta, reduce)
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
