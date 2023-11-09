from ..logger import get_module_logger

logger = get_module_logger("import_utils")


def is_torch_avaliable():
    try:
        import torch
    except ModuleNotFoundError as err:
        logger.warning("ModuleNotFoundError: torch is not installed, please install pytorch!")
        return False
    return True
