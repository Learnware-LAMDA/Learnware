# Name of the files used for checkpointing
import copy
import json
import logging
import os
from pathlib import Path

from ....config import Config

ROOT_PATH = Path(__file__).resolve().parent.parent
HETERO_ROOT_DIRPATH = os.path.join(ROOT_PATH, ".learnware")
PACKAGE_DIRPATH = os.path.dirname(os.path.abspath(__file__))

LEARNWARE_POOL_PATH = os.path.join(HETERO_ROOT_DIRPATH, "learnware_pool")
LEARNWARE_ZIP_POOL_PATH = os.path.join(LEARNWARE_POOL_PATH, "zips")
LEARNWARE_FOLDER_POOL_PATH = os.path.join(LEARNWARE_POOL_PATH, "learnwares")

DATABASE_PATH = os.path.join(HETERO_ROOT_DIRPATH, "database")
STDOUT_PATH = os.path.join(HETERO_ROOT_DIRPATH, "stdout")

# relative paths
TRAINING_ARGS_NAME = "training_args.json"
MODEL_PATH = "pytorch_model.bin"
TOKENIZER_DIR = "tokenizer"
HETERO_MAPPING_PATH = "hetero_mapping"

# TODO: Delete them later
# os.makedirs(HETERO_ROOT_DIRPATH, exist_ok=True)
# os.makedirs(DATABASE_PATH, exist_ok=True)
# os.makedirs(STDOUT_PATH, exist_ok=True)

_DEFAULT_CONFIG = {
    "root_path": HETERO_ROOT_DIRPATH,
    "package_path": PACKAGE_DIRPATH,
    "stdout_path": STDOUT_PATH,
    "logging_level": logging.INFO,
    "logging_outfile": None,
    "market_root_path": HETERO_ROOT_DIRPATH,
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
    "database_url": f"sqlite:///{DATABASE_PATH}"
}

C = Config(_DEFAULT_CONFIG)
