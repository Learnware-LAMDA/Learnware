import copy
from typing import Union

from ..model import BaseModel
from ..specification import BaseStatSpecification
from ..utils import get_module_by_module_path


def get_model_from_config(model: Union[BaseModel, dict]) -> BaseModel:
    """_summary_

    Parameters
    ----------
    model : Union[BaseModel, dict]
        - If isinstance(model, dict), model is must be the following format:
            model_dict = {
                "module_path": str, # path of python file
                "class_name": str, # the name of class in python file
            }
        - If isinstance(model, BaseModel), return model directly
    Returns
    -------
    BaseModel
        The model that is given by user
    Raises
    ------
    TypeError
        The type of model must be dict or BaseModel, else raise error
    """
    if isinstance(model, BaseModel):
        return model
    elif isinstance(model, dict):
        model_module = get_module_by_module_path(model["module_path"])
        return getattr(model_module, model["class_name"])(**model["kwargs"])
    else:
        raise TypeError("model must be type of BaseModel or str")


def get_stat_spec_from_config(stat_spec: dict) -> BaseStatSpecification:
    stat_spec_module = get_module_by_module_path(stat_spec["module_path"])
    stat_spec_inst = getattr(stat_spec_module, stat_spec["class_name"])(**stat_spec["kwargs"])

    if not isinstance(stat_spec_inst, BaseStatSpecification):
        raise TypeError(
            f"Statistic specification must be type of BaseStatSpecification, not {BaseStatSpecification.__class__.__name__}"
        )
    if stat_spec_inst.load(stat_spec["file_name"]) is False:
        raise ValueError("Load statistic specification failed!")

    return stat_spec_inst
