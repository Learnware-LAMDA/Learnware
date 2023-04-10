import os
import sys

import re
import yaml
import importlib
import importlib.util
from typing import Union
from types import ModuleType
from .logger import get_module_logger

logger = get_module_logger("utils")


def get_module_by_module_path(module_path: Union[str, ModuleType]):
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """save dict object into yaml file"""
    with open(save_path, "w") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


def read_yaml_to_dict(yaml_path: str):
    """load yaml file into dict object"""
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
