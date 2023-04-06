__version__ = "0.0.1.99"

from .logger import get_module_logger


def init(**kwargs):
    from .config import C

    C.update(**kwargs)

    logger = get_module_logger("Initialization")
    logger.info(f"init learnware market with {kwargs}")
