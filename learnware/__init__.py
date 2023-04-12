__version__ = "0.0.1.99"

import os
from .logger import get_module_logger


def init(reset=False, **kwargs):
    from .config import C

    C.reset()
    C.update(**kwargs)

    logger = get_module_logger("Initialization")

    ## make dirs
    os.makedirs(C.root_path, exist_ok=True)
    os.makedirs(C.database_path, exist_ok=True)
    os.makedirs(C.learnware_pool_path, exist_ok=True)
    os.makedirs(C.learnware_zip_pool_path, exist_ok=True)
    os.makedirs(C.learnware_folder_pool_path, exist_ok=True)

    logger.info(f"init learnware market with {kwargs}")
