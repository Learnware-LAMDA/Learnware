from ..logger import get_module_logger

logger = get_module_logger("import_utils")


def is_torch_avaliable():
    try:
        import torch
    except ModuleNotFoundError as err:
        logger.warning("ModuleNotFoundError: torch is not installed, please install pytorch!")
        return False
    return True


def is_lightgbm_avaliable():
    try:
        import lightgbm
    except ModuleNotFoundError as err:
        logger.warning("ModuleNotFoundError: lightgbm is not installed, please install lightgbm!")
        return False
    return True


def is_geatpy_avaliable():
    try:
        import geatpy
    except ModuleNotFoundError as err:
        logger.warning(
            "ModuleNotFoundError: geatpy is not installed, please install geatpy (only support python version<3.11)!"
        )
        return False
    return True
