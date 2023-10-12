import os
import numpy as np

import learnware
from learnware.client import LearnwareClient
from learnware.client.container import ModelEnvContainer, LearnwaresContainer
from learnware.learnware.reuse import AveragingReuser


if __name__ == "__main__":
    email = "liujd@lamda.nju.edu.cn"
    token = "f7e647146a314c6e8b4e2e1079c4bca4"
    
    client = LearnwareClient()
    client.login(email, token)
    
    learnware_ids = ["00000084", "00000154", "00000155"]
    zip_paths = ["1.zip", "2.zip", "3.zip"]
    root = os.path.dirname(__file__)
    for i in range(len(learnware_ids)):
        zip_paths[i] = os.path.join(root, zip_paths[i])
        client.download_learnware(learnware_ids[i], zip_paths[i])
    
    learnware_list = [client.load_learnware(file, load_model=False) for file in zip_paths]

    with LearnwaresContainer(learnware_list, zip_paths) as env_container:
        learnware_list = env_container.get_learnware_list_with_container()
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))
        
        for idx, learnware in enumerate(learnware_list):
            print(f"learnware_{idx}", reuser.predict(learnware))