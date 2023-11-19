__version__ = "0.2.0.1"

import os
import json
from .logger import get_module_logger
from .utils import is_torch_available, setup_seed

logger = get_module_logger("Initialization")


def init(**kwargs):
    """Init learnware package

    Parameters
    ----------
    deterministic : bool, optional
        whether to cancel randomness in learnware package, by default True
    mkdir : bool, optional
        whether to make directories for .learnware path, by default True
    tf_loglevel: str, optional
        The warning loglevel for tensforflow, by default "2"
    """
    from .config import C

    C.reset()

    config_file = os.path.join(C.root_path, "config.json")

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            C.update(**dict(json.load(f)))
    C.update(**{k: v for k, v in kwargs.items() if k in C})

    logger.info(f"init the learnware package with arguments {kwargs}")

    ## random seed
    deterministic = kwargs.get("deterministic", True)
    if deterministic:
        setup_seed(C.random_seed)

    ## make dirs
    mkdir = kwargs.get("mkdir", True)
    if mkdir:
        os.makedirs(C.root_path, exist_ok=True)
        os.makedirs(C.database_path, exist_ok=True)
        os.makedirs(C.stdout_path, exist_ok=True)
        os.makedirs(C.cache_path, exist_ok=True)

    ## ignore tensorflow warning
    tf_loglevel = kwargs.get("tf_loglevel", "2")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_loglevel


if not is_torch_available(verbose=False):
    logger.warning("The functionality of learnware is limited due to 'torch' is not installed!")

# default init package
init()
