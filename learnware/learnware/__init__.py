import os
import copy

from .base import Learnware, BaseReuser
from .reuse import JobSelectorReuser, AveragingReuser

from .utils import get_stat_spec_from_config, get_model_from_config
from ..specification import Specification
from ..utils import read_yaml_to_dict
from ..logger import get_module_logger
from ..config import C

logger = get_module_logger("learnware.learnware")


def get_learnware_from_dirpath(id: str, semantic_spec: dict, learnware_dirpath: str = None) -> Learnware:
    """Get the learnware object from dirpath, and provide the manage interface tor Learnware class

    Parameters
    ----------
    id : str
        The learnware id that is given by learnware market
    semantic_spec : dict
        The learnware semantice specifactions
    learnware_dirpath : str
        The dirpath of learnware folder

    Returns
    -------
    Learnware
        The contructed learnware object, return None if build failed
    """
    learnware_config = {
        "model": {
            "class_name": "Model",
            "kwargs": {},
        },
        "stat_specifications": [
            {
                "module_path": "learnware.specification",
                "class_name": "RKMEStatSpecification",
                "file_name": "stat_spec.json",
                "kwargs": {},
            },
        ],
    }

    if learnware_dirpath is not None:
        try:
            yaml_config = read_yaml_to_dict(os.path.join(learnware_dirpath, C.learnware_folder_config["yaml_file"]))
        except FileNotFoundError:
            yaml_config = {}

    if "name" in yaml_config:
        learnware_config["name"] = yaml_config["name"]
    if "model" in yaml_config:
        learnware_config["model"].update(yaml_config["model"])
    if "stat_specifications" in yaml_config:
        learnware_config["stat_specifications"] = yaml_config["stat_specifications"].copy()

    if "module_path" not in learnware_config["model"]:
        learnware_config["model"]["module_path"] = os.path.join(
            learnware_dirpath, C.learnware_folder_config["module_file"]
        )

    try:
        learnware_spec = Specification()
        for _stat_spec in learnware_config["stat_specifications"]:
            stat_spec = _stat_spec.copy()
            stat_spec["file_name"] = os.path.join(learnware_dirpath, stat_spec["file_name"])
            stat_spac_name, stat_spec_inst = get_stat_spec_from_config(stat_spec)
            learnware_spec.update_stat_spec(**{stat_spac_name: stat_spec_inst})

        learnware_spec.upload_semantic_spec(copy.deepcopy(semantic_spec))
        # learnware_model = get_model_from_config(learnware_config["model"])

    except Exception as e:
        logger.warning(f"Load Learnware {id} failed! Due to {repr(e)}")
        return None

    return Learnware(id=id, model=learnware_config["model"], specification=learnware_spec)
