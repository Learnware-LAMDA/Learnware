import logging
from logging import Logger, handlers

from .config import C


def get_module_logger(module_name: str, level: int = None, outfile: str = None) -> Logger:
    """Get a logger for a specific module.
    Parameters
    ----------
    module_name : str
        Logic module name.
    level : int, optional
        Logging level, by default None
    outfile : str, optional
        The output filepath, by default None

    Returns
    -------
    Logger
        logging.Logger for output log
    """
    if level is None:
        level = C.logging_level
    if outfile is None:
        outfile = C.logging_outfile

    # Get logger.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(f"[%(levelname)s] - %(asctime)s - %(filename)s - {module_name} - %(message)s")
    console_handler.setFormatter(formatter)
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(level)
    module_logger.addHandler(console_handler)

    if outfile is not None:
        file_handler = logging.FileHandler(outfile)
        file_handler.setFormatter(formatter)
        module_logger.addHandler(file_handler)

    return module_logger
