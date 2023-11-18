import numpy as np
from ..logger import get_module_logger

logger = get_module_logger("reuse_utils")


def is_geatpy_available(verbose=False):
    try:
        import geatpy
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning(
                "ModuleNotFoundError: geatpy is not installed, please install geatpy (only support python version<3.11)!"
            )
        return False
    return True


def is_lightgbm_available(verbose=False):
    try:
        import lightgbm
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning("ModuleNotFoundError: lightgbm is not installed, please install lightgbm!")
        return False
    return True


def fill_data_with_mean(X: np.ndarray) -> np.ndarray:
    """
    Fill missing data (NaN, Inf) in the input array with the mean of the column.

    Parameters
    ----------
    X : np.ndarray
        Input data array that may contain missing values.

    Returns
    -------
    np.ndarray
        Data array with missing values filled.

    Raises
    ------
    ValueError
        If a column in X contains only exceptional values (NaN, Inf).
    """
    X[np.isinf(X) | np.isneginf(X) | np.isposinf(X) | np.isneginf(X)] = np.nan
    if np.any(np.isnan(X)):
        for col in range(X.shape[1]):
            is_nan = np.isnan(X[:, col])
            if np.any(is_nan):
                if np.all(is_nan):
                    raise ValueError(f"All values in column {col} are exceptional, e.g., NaN and Inf.")
                col_mean = np.nanmean(X[:, col])
                X[:, col] = np.where(is_nan, col_mean, X[:, col])
    return X
