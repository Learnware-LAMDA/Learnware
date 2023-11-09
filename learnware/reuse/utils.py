from ..logger import get_module_logger

logger = get_module_logger("reuse_utils")


def is_geatpy_avaliable(verbose=False):
    try:
        import geatpy
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning(
                "ModuleNotFoundError: geatpy is not installed, please install geatpy (only support python version<3.11)!"
            )
        return False
    return True


def is_lightgbm_avaliable(verbose=False):
    try:
        import lightgbm
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning("ModuleNotFoundError: lightgbm is not installed, please install lightgbm!")
        return False
    return True
