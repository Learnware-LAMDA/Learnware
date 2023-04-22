import os
import copy
import logging


class Config:
    def __init__(self, default_conf):
        self.__dict__["_default_config"] = copy.deepcopy(default_conf)  # avoiding conflictions with __getattr__
        self.reset()

    def __getitem__(self, key):
        return self.__dict__["_config"][key]

    def __getattr__(self, attr):
        if attr in self.__dict__["_config"]:
            return self.__dict__["_config"][attr]

        raise AttributeError(f"No such {attr} in self._config")

    def get(self, key, default=None):
        return self.__dict__["_config"].get(key, default)

    def __setitem__(self, key, value):
        self.__dict__["_config"][key] = value

    def __setattr__(self, attr, value):
        self.__dict__["_config"][attr] = value

    def __contains__(self, item):
        return item in self.__dict__["_config"]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_config"])

    def __repr__(self):
        return str(self.__dict__["_config"])

    def reset(self):
        self.__dict__["_config"] = copy.deepcopy(self._default_config)

    def update(self, *args, **kwargs):
        self.__dict__["_config"].update(*args, **kwargs)


ROOT_DIRPATH = os.path.join(os.path.expanduser("~"), ".learnware")
SPEC_DIRPATH = None

LEARNWARE_POOL_PATH = os.path.join(ROOT_DIRPATH, "learnware_pool")
LEARNWARE_ZIP_POOL_PATH = os.path.join(LEARNWARE_POOL_PATH, "zips")
LEARNWARE_FOLDER_POOL_PATH = os.path.join(LEARNWARE_POOL_PATH, "learnwares")

DATABASE_PATH = os.path.join(ROOT_DIRPATH, "database")


# TODO: Delete them later
os.makedirs(ROOT_DIRPATH, exist_ok=True)
os.makedirs(DATABASE_PATH, exist_ok=True)

semantic_config = {
    "Data": {
        "Values": ["Table", "Image", "Video", "Text", "Audio"],
        "Type": "Class",
    },  # Choose only one class
    "Task": {
        "Values": [
            "Classification",
            "Regression",
            "Clustering",
            "Feature Extraction",
            # "Generation",
            "Segmentation",
            "Object Detection",
            "Others",
        ],
        "Type": "Class",  # Choose only one class
    },
    # "Device": {
    #     "Values": ["CPU", "GPU"],
    #     "Type": "Tag",
    # },  # Choose one or more tags
    "Library": {
        "Values": ["Scikit-learn", "PyTorch", "TensorFlow", "Others"],
        "Type": "Class",
    },  # Choose one or more tags
    "Scenario": {
        "Values": [
            "Business",
            "Financial",
            "Health",
            "Politics",
            "Computer",
            "Internet",
            "Traffic",
            "Nature",
            "Fashion",
            "Industry",
            "Agriculture",
            "Education",
            "Entertainment",
            "Architecture",
        ],
        "Type": "Tag",  # Choose one or more tags
    },
    "Description": {
        "Values": None,
        "Type": "String",
    },
    "Name": {
        "Values": None,
        "Type": "String",
    },
}

_DEFAULT_CONFIG = {
    "root_path": ROOT_DIRPATH,
    "logging_level": logging.INFO,
    "logging_outfile": None,
    "semantic_specs": semantic_config,
    "market_root_path": ROOT_DIRPATH,
    "learnware_pool_path": LEARNWARE_POOL_PATH,
    "learnware_zip_pool_path": LEARNWARE_ZIP_POOL_PATH,
    "learnware_folder_pool_path": LEARNWARE_FOLDER_POOL_PATH,
    "learnware_folder_config": {
        "yaml_file": "learnware.yaml",
        "module_file": "__init__.py",
    },
    "database_path": DATABASE_PATH,
    "max_reduced_set_size": 1310720,
}

C = Config(_DEFAULT_CONFIG)
