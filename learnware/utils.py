import os
from .logger import get_module_logger

logger = get_module_logger("utils")


def make_dir_by_path(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        logger.warning(f"Directorty {dirpath} has been exited, ignore mkdir")
