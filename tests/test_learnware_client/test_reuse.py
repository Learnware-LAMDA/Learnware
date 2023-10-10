from learnware.learnware import get_learnware_from_dirpath, Learnware
from learnware.market import EasyMarket
from learnware.client.container import ModelEnvContainer

if __name__ == "__main__":
    semantic_specification = dict()
    semantic_specification["Data"] = {"Type": "Class", "Values": ["Text"]}
    semantic_specification["Task"] = {"Type": "Class", "Values": ["Ranking"]}
    semantic_specification["Library"] = {"Type": "Class", "Values": ["Scikit-learn"]}
    semantic_specification["Scenario"] = {"Type": "Tag", "Values": "Financial"}
    semantic_specification["Name"] = {"Type": "String", "Values": "test"}
    semantic_specification["Description"] = {"Type": "String", "Values": "test"}
    
    zip_path = '/home/bixd/workspace/learnware/Learnware/tests/test_workflow/learnware_pool/svm_0.zip'
    
    learnware = get_learnware_from_dirpath('test_id', semantic_specification, zip_path)
    
    env_leanware = Learnware(id=learnware.id, model=ModelEnvContainer(learnware.get_model(), zip_path), specification=learnware.get_specification())
    
    print('check', EasyMarket.check_learnware(env_leanware))
    