import zipfile
import numpy as np

from learnware.learnware import get_learnware_from_dirpath, Learnware
from learnware.market import EasyMarket
from learnware.client.container import ModelEnvContainer
from learnware.learnware.reuse import AveragingReuser

if __name__ == "__main__":
    semantic_specification = dict()
    semantic_specification["Data"] = {"Type": "Class", "Values": ["Text"]}
    semantic_specification["Task"] = {"Type": "Class", "Values": ["Ranking"]}
    semantic_specification["Library"] = {"Type": "Class", "Values": ["Scikit-learn"]}
    semantic_specification["Scenario"] = {"Type": "Tag", "Values": "Financial"}
    semantic_specification["Name"] = {"Type": "String", "Values": "test"}
    semantic_specification["Description"] = {"Type": "String", "Values": "test"}
    
    zip_paths = [
        '/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/rf_tic.zip',
        '/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/svc_tic.zip',
    ]
    dir_paths = [
        '/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/rf_tic',
        '/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/svc_tic',
    ]
    
    learnware_list = []
    for id, (zip_path, dir_path) in enumerate(zip(zip_paths, dir_paths)):
        with zipfile.ZipFile(zip_path, "r") as z_file:
            z_file.extractall(dir_path)
            
        learnware = get_learnware_from_dirpath(f'test_id{id}', semantic_specification, dir_path)

        model = ModelEnvContainer(learnware.get_model(), zip_path)
        model.init_env_and_metadata()
        
        env_leanware = Learnware(id=learnware.id, model=model, specification=learnware.get_specification())
            
        learnware_list.append(env_leanware)
        
        print('check:', EasyMarket.check_learnware(env_leanware))
    
    reuser = AveragingReuser(learnware_list, mode='vote')
    input_array = np.random.randint(0, 3, size=(20, 9))
    print(reuser.predict(input_array).argmax(axis=1))
    
    for id, ind_learner in enumerate(learnware_list):
        print(f"learner_{id}", reuser.predict(input_array).argmax(axis=1))
    
    