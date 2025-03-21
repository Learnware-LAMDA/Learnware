def parse_specification_type(
    stat_specs: dict,
    spec_list=[
        "HeteroMapTableSpecification",
        "RKMETableSpecification",
        "GenerativeModelSpecification",
        "RKMETextSpecification",
        "RKMEImageSpecification",
        "LLMGeneralCapabilitySpecification",
    ],
):
    for spec in spec_list:
        if spec in stat_specs:
            return spec
    return None
