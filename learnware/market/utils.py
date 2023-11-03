from ..specification import Specification


def parse_specification_type(
    stat_spec: Specification, spec_list=["RKMETableSpecification", "RKMETextSpecification", "RKMEImageSpecification"]
):
    stat_specs = stat_spec.stat_spec
    for spec in spec_list:
        if spec in stat_specs:
            return spec
    return None
