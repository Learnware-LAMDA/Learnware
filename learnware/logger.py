import logging
import logging.handlers

from .config import C


def get_module_logger(module_name: str, level:int = None):
    """Get a logger for a specific module.

    Parameters
    ----------
    module_name : str
        Logic module name.
    level : int, optional
        logging level, by default None

    Returns
    -------
    _type_
        _description_
    """
    if level is None:
        level = C.logging_level

    # Get logger.
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(level)
    return module_logger
