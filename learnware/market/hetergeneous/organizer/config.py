# Name of the files used for checkpointing
import copy
import json
import logging
import os

from ....config import Config

ROOT_PATH = os.path.join(os.path.expanduser("~"), ".learnware")
HETERO_ROOT_PATH = os.path.join(ROOT_PATH, "heterogeneous")
PACKAGE_DIRPATH = os.path.dirname(os.path.abspath(__file__))

LEARNWARE_POOL_PATH = os.path.join(HETERO_ROOT_PATH, "learnware_pool")
LEARNWARE_ZIP_POOL_PATH = os.path.join(LEARNWARE_POOL_PATH, "zips")
LEARNWARE_FOLDER_POOL_PATH = os.path.join(LEARNWARE_POOL_PATH, "learnwares")

DATABASE_PATH = os.path.join(HETERO_ROOT_PATH, "database")
STDOUT_PATH = os.path.join(HETERO_ROOT_PATH, "stdout")

# relative paths
TRAINING_ARGS_NAME = "training_args.json"
MODEL_PATH = "pytorch_model.bin"
TOKENIZER_DIR = "tokenizer"
HETERO_MAPPING_PATH = "hetero_mappings"

# TODO: Delete them later
# os.makedirs(HETERO_ROOT_DIRPATH, exist_ok=True)
# os.makedirs(DATABASE_PATH, exist_ok=True)
# os.makedirs(STDOUT_PATH, exist_ok=True)

_DEFAULT_CONFIG = {
    "root_path": ROOT_PATH,
    "hetero_root_path": HETERO_ROOT_PATH,
    "package_path": PACKAGE_DIRPATH,
    "stdout_path": STDOUT_PATH,
    "logging_level": logging.INFO,
    "logging_outfile": None,
    "market_model_path": MODEL_PATH,
    "market_training_args_path": TRAINING_ARGS_NAME,
    "market_tokenizer_path": TOKENIZER_DIR,
    "heter_mapping_path": HETERO_MAPPING_PATH,
    "learnware_pool_path": LEARNWARE_POOL_PATH,
    "learnware_zip_pool_path": LEARNWARE_ZIP_POOL_PATH,
    "learnware_folder_pool_path": LEARNWARE_FOLDER_POOL_PATH,
    "learnware_folder_config": {
        "yaml_file": "learnware.yaml",
        "module_file": "__init__.py",
    },
    "database_url": f"sqlite:///{DATABASE_PATH}",
}

C = Config(_DEFAULT_CONFIG)
