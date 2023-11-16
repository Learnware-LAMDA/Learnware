from ..specification import Specification


def parse_specification_type(
    stat_specs: dict,
    spec_list=[
        "HeteroMapTableSpecification",
        "RKMETableSpecification",
        "RKMETextSpecification",
        "RKMEImageSpecification",
    ],
):
    for spec in spec_list:
        if spec in stat_specs:
            return spec
    return None
