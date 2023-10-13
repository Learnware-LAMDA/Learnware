import os
import unittest
import zipfile
import numpy as np

import learnware
from learnware.learnware import get_learnware_from_dirpath
from learnware.client import LearnwareClient
from learnware.client.container import ModelEnvContainer, LearnwaresContainer
from learnware.learnware.reuse import AveragingReuser


class TestLearnwareLoad(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUpClass()
        email = "liujd@lamda.nju.edu.cn"
        token = "f7e647146a314c6e8b4e2e1079c4bca4"

        self.client = LearnwareClient()
        self.client.login(email, token)

        learnware_ids = ["00000084", "00000154", "00000155"]
        zip_paths = ["1.zip", "2.zip", "3.zip"]
        root = os.path.dirname(__file__)
        for i in range(len(learnware_ids)):
            zip_paths[i] = os.path.join(root, zip_paths[i])
            self.client.download_learnware(learnware_ids[i], zip_paths[i])
        self.zip_paths = zip_paths

    def test_single_learnware(self):
        learnware_list = [self.client.load_learnware(zippath, load_option="conda_env") for zippath in self.zip_paths]
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))

    def test_multi_learnware(self):
        learnware_list = self.client.load_learnware(self.zip_paths, load_option="conda_env")
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))


if __name__ == '__main__':
    unittest.main()