import os
import unittest
import argparse
import numpy as np

import learnware
from learnware.learnware import get_learnware_from_dirpath
from learnware.client import LearnwareClient
from learnware.client.container import ModelCondaContainer, LearnwaresContainer
from learnware.reuse import AveragingReuser


class TestLearnwareLoadWithConda(unittest.TestCase):
    def setUp(self):
        self.client = LearnwareClient()
        root = os.path.dirname(__file__)
        self.learnware_ids = ["00000084", "00000154", "00000155"]
        self.zip_paths = [os.path.join(root, x) for x in ["1.zip", "2.zip", "3.zip"]]
        self.runnable_option = "conda"

    #def test_load_single_learnware_by_zippath(self):
    #    for learnware_id, zip_path in zip(self.learnware_ids, self.zip_paths):
    #        self.client.download_learnware(learnware_id, zip_path)
#
    #    learnware_list = [
    #        self.client.load_learnware(learnware_path=zippath, runnable_option=self.runnable_option) for zippath in self.zip_paths
    #    ]
    #    reuser = AveragingReuser(learnware_list, mode="vote_by_label")
    #    input_array = np.random.random(size=(20, 13))
    #    print(reuser.predict(input_array))
#
    #    for learnware in learnware_list:
    #        print(learnware.id, learnware.predict(input_array))
#
    #def test_load_multi_learnware_by_zippath(self):
    #    for learnware_id, zip_path in zip(self.learnware_ids, self.zip_paths):
    #        self.client.download_learnware(learnware_id, zip_path)
#
    #    learnware_list = self.client.load_learnware(learnware_path=self.zip_paths, runnable_option=self.runnable_option)
    #    reuser = AveragingReuser(learnware_list, mode="vote_by_label")
    #    input_array = np.random.random(size=(20, 13))
    #    print(reuser.predict(input_array))
#
    #    for learnware in learnware_list:
    #        print(learnware.id, learnware.predict(input_array))
#
    #def test_load_single_learnware_by_id(self):
    #    learnware_list = [
    #        self.client.load_learnware(learnware_id=idx, runnable_option=self.runnable_option) for idx in self.learnware_ids
    #    ]
    #    reuser = AveragingReuser(learnware_list, mode="vote_by_label")
    #    input_array = np.random.random(size=(20, 13))
    #    print(reuser.predict(input_array))
#
    #    for learnware in learnware_list:
    #        print(learnware.id, learnware.predict(input_array))
#
    #def test_load_multi_learnware_by_id(self):
    #    learnware_list = self.client.load_learnware(learnware_id=self.learnware_ids, runnable_option=self.runnable_option)
    #    reuser = AveragingReuser(learnware_list, mode="vote_by_label")
    #    input_array = np.random.random(size=(20, 13))
    #    print(reuser.predict(input_array))
#
    #    for learnware in learnware_list:
    #        print(learnware.id, learnware.predict(input_array))
#
    def test_load_single_learnware_by_id_pip(self):
        learnware_id = "00000147"
        learnware = self.client.load_learnware(learnware_id=learnware_id, runnable_option=self.runnable_option)
        input_array = np.random.random(size=(20, 23))
        print(learnware.predict(input_array))
#
    def test_load_single_learnware_by_id_conda(self):
        learnware_id = "00000148"
        learnware = self.client.load_learnware(learnware_id=learnware_id, runnable_option=self.runnable_option)
        input_array = np.random.random(size=(20, 204))
        print(learnware.predict(input_array))
#
#
class TestLearnwareLoadWithDocker(TestLearnwareLoadWithConda):
    def setUp(self):
        super(TestLearnwareLoadWithDocker, self).setUp()
        self.runnable_option = "docker"

def suite(mode):
    _suite = unittest.TestSuite()
    #_suite.addTest(TestLearnwareLoadWithDocker())
    if mode == "all" or mode == "conda":
        _suite.addTest(unittest.makeSuite(TestLearnwareLoadWithConda))
    if mode == "all" or mode == "docker":
        _suite.addTest(unittest.makeSuite(TestLearnwareLoadWithDocker))
    return _suite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="all", help="The mode to run load learnware, must be in ['all', 'conda', 'docker']")
    args = parser.parse_args()

    assert args.mode in {"all", "conda", "docker"}, f"The mode must be in ['all', 'conda', 'docker'], instead of '{args.mode}'"
    runner = unittest.TextTestRunner()
    runner.run(suite(args.mode))