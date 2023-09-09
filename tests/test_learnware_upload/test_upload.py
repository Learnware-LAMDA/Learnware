import os

import learnware
from learnware.client.learnware_client import LearnwareClient


if __name__ == "__main__":
    semantic_specification = dict()
    semantic_specification["Data"] = {"Type": "Class", "Values": ["Text"]}
    semantic_specification["Task"] = {"Type": "Class", "Values": ["Ranking"]}
    semantic_specification["Library"] = {"Type": "Class", "Values": ["Scikit-learn"]}
    semantic_specification["Scenario"] = {"Type": "Tag", "Values": "Financial"}
    semantic_specification["Name"] = {"Type": "String", "Values": "test"}
    semantic_specification["Description"] = {"Type": "String", "Values": "test"}
    
    zip_path = "test.zip"
    client = LearnwareClient()
    client.install_environment(zip_path)
    client.test_learnware(zip_path, semantic_specification)