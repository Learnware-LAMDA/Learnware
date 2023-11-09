__version__ = "0.1.1.99"

import os
from .logger import get_module_logger


def init(make_dir: bool = False, tf_loglevel: str = "2", **kwargs):
    from .config import C

    C.reset()
    C.update(**kwargs)

    logger = get_module_logger("Initialization")
    logger.info(f"init learnware market with {kwargs}")

    ## make dirs
    if make_dir:
        os.makedirs(C.root_path, exist_ok=True)
        os.makedirs(C.database_path, exist_ok=True)
        os.makedirs(C.learnware_pool_path, exist_ok=True)
        os.makedirs(C.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(C.learnware_folder_pool_path, exist_ok=True)
        logger.info(f"make learnware dir successfully!")

    ## ignore tensorflow warning
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_loglevel
    # logger.info(f"The tensorflow log level is setted to {tf_loglevel}")
