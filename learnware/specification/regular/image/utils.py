from ....logger import get_module_logger

logger = get_module_logger("regular_image_spec_utils")


def is_torch_optimizer_available(verbose=False):
    try:
        import torch_optimizer
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning("ModuleNotFoundError: torch_optimizer is not installed, please install torch_optimizer!")
        return False
    return True


def is_torchvision_available(verbose=False):
    try:
        import torchvision
    except ModuleNotFoundError as err:
        if verbose is True:
            logger.warning("ModuleNotFoundError: torchvision is not installed, please install torchvision!")
        return False
    return True
