def get_semantic_specification():
    semantic_specification = dict()
    semantic_specification["Data"] = {"Type": "Class", "Values": ["Text"]}
    semantic_specification["Task"] = {"Type": "Class", "Values": ["Segmentation"]}
    semantic_specification["Library"] = {"Type": "Class", "Values": ["Scikit-learn"]}
    semantic_specification["Scenario"] = {"Type": "Tag", "Values": ["Financial"]}
    semantic_specification["Name"] = {"Type": "String", "Values": "test"}
    semantic_specification["Description"] = {"Type": "String", "Values": "test"}
    return semantic_specification
