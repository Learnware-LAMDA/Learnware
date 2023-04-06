import logging
from logging import Logger, handlers

from .config import C


def get_module_logger(module_name: str, level:int = None) -> Logger:
    """Get a logger for a specific module.
    Parameters
    ----------
    module_name : str
        Logic module name.
    level : int, optional
        Logging level, by default None

    Returns
    -------
    _type_
        _description_
    """
    if level is None:
        level = C.logging_level

    # Get logger.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(level)
    module_logger.addHandler(console_handler)
    return module_logger
