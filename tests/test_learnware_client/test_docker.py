import os
import zipfile
import numpy as np

import learnware
from learnware.client import LearnwareClient
from learnware.client.container import LearnwaresContainer
from learnware.reuse import AveragingReuser


if __name__ == "__main__":
    email = "liujd@lamda.nju.edu.cn"
    token = "f7e647146a314c6e8b4e2e1079c4bca4"

    client = LearnwareClient()
    client.login(email, token)

    root = os.path.dirname(__file__)
    learnware_ids = ["00000084", "00000154", "00000155"]
    zip_paths = [os.path.join(root, x) for x in ["1.zip", "2.zip", "3.zip"]]

    for learnware_id, zip_path in zip(learnware_ids, zip_paths):
        client.download_learnware(learnware_id, zip_path)

    learnware_list = [client.load_learnware(learnware_path=zippath) for zippath in zip_paths]
    with LearnwaresContainer(learnware_list, zip_paths, mode="docker") as env_container:
        learnware_list = env_container.get_learnwares_with_container()
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))
