from ..logger import get_module_logger

logger = get_module_logger("import_utils")


def is_torch_available(verbose=False):
    try:
        import torch
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning("ModuleNotFoundError: torch is not installed, please install pytorch!")
        return False
    return True
