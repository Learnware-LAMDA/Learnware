import os
import unittest
import zipfile
import numpy as np

import learnware
from learnware.learnware import get_learnware_from_dirpath
from learnware.client import LearnwareClient
from learnware.client.container import ModelCondaContainer, LearnwaresContainer
from learnware.learnware.reuse import AveragingReuser


class TestLearnwareLoad(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        email = "liujd@lamda.nju.edu.cn"
        token = "f7e647146a314c6e8b4e2e1079c4bca4"

        self.client = LearnwareClient()
        self.client.login(email, token)

        root = os.path.dirname(__file__)
        self.learnware_ids = ["00000084", "00000154", "00000155"]
        self.zip_paths = [os.path.join(root, x) for x in ["1.zip", "2.zip", "3.zip"]]

    def test_load_multi_learnware_by_zippath(self):
        for learnware_id, zip_path in zip(self.learnware_ids, self.zip_paths):
            self.client.download_learnware(learnware_id, zip_path)

        learnware_list = self.client.load_learnware(learnware_path=self.zip_paths, runnable_option="docker")
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))

    def test_load_multi_learnware_by_id(self):
        learnware_list = self.client.load_learnware(learnware_id=self.learnware_ids, runnable_option="docker")
        docker_container = learnware_list[0].get_model().docker_container
        
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))
        
        learnware_list[0].get_model()._destroy_docker_container(docker_container)


if __name__ == "__main__":
    unittest.main()