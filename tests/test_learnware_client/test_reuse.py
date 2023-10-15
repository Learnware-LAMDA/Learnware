import zipfile
import numpy as np

from learnware.learnware import get_learnware_from_dirpath
from learnware.client.container import LearnwaresContainer
from learnware.learnware.reuse import AveragingReuser
from learnware.test.module import get_semantic_specification

if __name__ == "__main__":
    semantic_specification = get_semantic_specification()
    zip_paths = [
        "/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/rf_tic.zip",
        "/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/svc_tic.zip",
    ]
    dir_paths = [
        "/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/rf_tic",
        "/home/bixd/workspace/learnware/Learnware/tests/test_learnware_client/svc_tic",
    ]   

    learnware_list = []
    for id, (zip_path, dir_path) in enumerate(zip(zip_paths, dir_paths)):
        with zipfile.ZipFile(zip_path, "r") as z_file:
            z_file.extractall(dir_path)

        learnware = get_learnware_from_dirpath(f"test_id{id}", semantic_specification, dir_path)
        learnware_list.append(learnware)

    env_container = LearnwaresContainer(learnware_list, zip_paths)
    learnware_list = env_container.get_learnwares_with_container()
    reuser = AveragingReuser(learnware_list, mode="vote")
    input_array = np.random.randint(0, 3, size=(20, 9))
    print(reuser.predict(input_array).argmax(axis=1))
    for id, ind_learner in enumerate(learnware_list):
        print(f"learner_{id}", reuser.predict(input_array).argmax(axis=1))
