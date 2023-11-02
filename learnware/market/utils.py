from ..specification import Specification
from ..logger import get_module_logger

logger = get_module_logger("market_utils")


def parse_specification_type(stat_spec: Specification):
    stat_specs = stat_spec.stat_spec
    spec_list = ["RKMETableSpecification", "RKMETextSpecification", "RKMEImageSpecification"]
    for spec in spec_list:
        if spec in stat_specs:
            return spec
    return None
