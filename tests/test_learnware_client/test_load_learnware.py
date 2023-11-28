import os
import unittest
import argparse
import numpy as np

import learnware
from learnware.learnware import get_learnware_from_dirpath
from learnware.client import LearnwareClient
from learnware.client.container import ModelCondaContainer, LearnwaresContainer
from learnware.reuse import AveragingReuser
from learnware.tests import parametrize


class TestLearnwareLoad(unittest.TestCase):
    def __init__(self, method_name='runTest', mode="conda"):
        super(TestLearnwareLoad, self).__init__(method_name)
        self.runnable_options = []
        if mode in {"all", "conda"}:
            self.runnable_options.append("conda")
        if mode in {"all", "docker"}:
            self.runnable_options.append("docker")

    def setUp(self):
        self.client = LearnwareClient()
        root = os.path.dirname(__file__)
        self.learnware_ids = ["00000084", "00000154", "00000155"]
        self.zip_paths = [os.path.join(root, x) for x in ["1.zip", "2.zip", "3.zip"]]

    def _test_load_learnware_by_zippath(self, runnable_option):
        for learnware_id, zip_path in zip(self.learnware_ids, self.zip_paths):
            self.client.download_learnware(learnware_id, zip_path)

        learnware_list = self.client.load_learnware(learnware_path=self.zip_paths, runnable_option=runnable_option)
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))
        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))


    def _test_load_learnware_by_id(self, runnable_option):
        learnware_list = self.client.load_learnware(learnware_id=self.learnware_ids, runnable_option=runnable_option)
        reuser = AveragingReuser(learnware_list, mode="vote_by_label")
        input_array = np.random.random(size=(20, 13))
        print(reuser.predict(input_array))

        for learnware in learnware_list:
            print(learnware.id, learnware.predict(input_array))

    def test_load_learnware_by_zippath(self):
        for runnable_option in self.runnable_options:
            self._test_load_learnware_by_zippath(runnable_option=runnable_option)
    
    def test_load_learnware_by_id(self):
        for runnable_option in self.runnable_options:
            self._test_load_learnware_by_id(runnable_option=runnable_option)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="conda", help="The mode to load learnware, must be in ['all', 'conda', 'docker']")
    args = parser.parse_args()

    assert args.mode in {"all", "conda", "docker"}, f"The mode must be in ['all', 'conda', 'docker'], instead of '{args.mode}'"
    runner = unittest.TextTestRunner()
    runner.run(parametrize(TestLearnwareLoad, mode=args.mode))