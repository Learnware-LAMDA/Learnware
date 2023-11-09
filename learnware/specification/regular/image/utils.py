from ....logger import get_module_logger

logger = get_module_logger("regular_image_spec_utils")


def is_torch_optimizer_avaliable():
    try:
        import torch_optimizer
    except ModuleNotFoundError as err:
        logger.warning("ModuleNotFoundError: torch_optimizer is not installed, please install pytorch!")
        return False
    return True
