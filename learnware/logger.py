import logging
import logging.handlers

from .config import C


def get_module_logger(module_name, level=logging.INFO):
    """
    Get a logger for a specific module.
    :param module_name: str
        Logic module name.
    :param level: int
    :param sh_level: int
        Stream handler log level.
    :param log_format: str
    :return: Logger
        Logger object.
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
