from .base import Learnware
from .utils import get_stat_spec_from_config, get_model_from_config
from ..specification import RKMEStatSpecification, Specification
from ..utils import get_module_by_module_path
from ..logger import get_module_logger

from typing import Tuple

from .base import Learnware

logger = get_module_logger("learnware.learnware")


def get_learnware_from_config(id: int, file_config: dict, semantic_spec: dict) -> Learnware:
    """Get the learnware object from config, and provide the manage interface tor Learnware class

    Parameters
    ----------
    id : int
        The learnware id that is given by learnware market
    file_config : dict
        The learnware file config that demonstrates the name, model, and statistic specification config of learnware
    semantic_spec : dict
        The learnware semantice specifactions

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
    if "name" in file_config:
        learnware_config["name"] = file_config["name"]
    if "model" in file_config:
        learnware_config["model"].update(file_config["model"])
    if "stats_specifications" in file_config:
        learnware_config["stat_specifications"] = file_config["stat_specifications"]

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
