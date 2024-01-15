import os
import unittest

import numpy as np

from learnware.client import LearnwareClient
from learnware.reuse import AveragingReuser


class TestLearnwareLoad(unittest.TestCase):
    def __init__(self, method_name="runTest", mode="all"):
        super(TestLearnwareLoad, self).__init__(method_name)
        self.runnable_options = []
        if mode in {"all", "conda"}:
            self.runnable_options.append("conda")
        if mode in {"all", "docker"}:
            self.runnable_options.append("docker")

    def setUp(self):
        self.client = LearnwareClient()
        root = os.path.dirname(__file__)
        self.learnware_ids = ["00000910", "00000899", "00000900"]
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


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestLearnwareLoad("test_load_learnware_by_zippath", mode="all"))
    _suite.addTest(TestLearnwareLoad("test_load_learnware_by_id", mode="all"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
