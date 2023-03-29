import numpy as np
from .base import StatSpecification


def generate_stat_spec(X: np.ndarray) -> StatSpecification:
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
