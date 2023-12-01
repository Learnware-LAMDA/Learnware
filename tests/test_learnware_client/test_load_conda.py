import os
import unittest
import zipfile
import numpy as np

import learnware
from learnware.learnware import get_learnware_from_dirpath
from learnware.client import LearnwareClient
from learnware.client.container import ModelCondaContainer, LearnwaresContainer
from learnware.reuse import AveragingReuser


class TestLearnwareLoad(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        self.client = LearnwareClient()

        root = os.path.dirname(__file__)
        self.learnware_ids = ["00000910", "00000899", "00000900"]
        self.zip_paths = [os.path.join(root, x) for x in ["1.zip", "2.zip", "3.zip"]]

    def test_load_multi_learnware_by_zippath(self):
        for learnware_id, zip_path in zip(self.learnware_ids, self.zip_paths):
            self.client.download_learnware(learnware_id, zip_path)

        learnware_list = self.client.load_learnware(learnware_path=self.zip_paths, runnable_option="conda")
        reuser = AveragingReuser(learnware_list, mode="mean")
        input_array = np.random.random(size=(20, 40))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))



if __name__ == "__main__":
    unittest.main()
