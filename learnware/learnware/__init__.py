import os

from .base import Learnware
from .utils import get_stat_spec_from_config, get_model_from_config
from ..specification import Specification
from ..utils import read_yaml_to_dict
from ..logger import get_module_logger

logger = get_module_logger("learnware.learnware")


def get_learnware_from_dirpath(id: int, semantic_spec: dict, learnware_dirpath: str = None) -> Learnware:
    """Get the learnware object from dirpath, and provide the manage interface tor Learnware class

    Parameters
    ----------
    id : int
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
        "name": "None",
        "model": {
            "class_name": "Model",
            "kwargs": {},
        },
        "stat_specifications": [
            {
                "module_name": "learnware.specification",
                "class_name": "RKMEStatSpecification",
                "kwargs": {},
            },
        ],
    }

    if learnware_dirpath is not None:
        try:
            yaml_config = read_yaml_to_dict(os.path.join(learnware_dirpath, "learnware.yaml"))
        except FileNotFoundError:
            yaml_config = {}

    if "name" in yaml_config:
        learnware_config["name"] = yaml_config["name"]
    if "model" in yaml_config:
        learnware_config["model"].update(yaml_config["model"])
    if "stats_specifications" in yaml_config:
        learnware_config["stat_specifications"] = yaml_config["stat_specifications"]

    try:
        learnware_spec = Specification()
        for _stat_spec in learnware_config["stat_specifications"]:
            stat_spac_name, stat_spec_inst = get_stat_spec_from_config(_stat_spec)
            learnware_spec.update_stat_spec(**{stat_spac_name: stat_spec_inst})

        learnware_spec.upload_semantic_spec(semantic_spec)
        learnware_model = get_model_from_config(learnware_config["model"])

    except Exception:
        logger.warning(f"Load Learnware {id} failed!")
        return None

    return Learnware(id=id, name=learnware_config["name"], model=learnware_model, specification=learnware_spec)
