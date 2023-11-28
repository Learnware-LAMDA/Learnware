import os
import unittest
import argparse
import numpy as np

from learnware.learnware import get_learnware_from_dirpath
from learnware.client import LearnwareClient
from learnware.client.container import ModelCondaContainer, LearnwaresContainer
from learnware.tests import parametrize

class TestContainer(unittest.TestCase):
    def __init__(self, method_name='runTest', mode="all"):
        super(TestContainer, self).__init__(method_name)
        self.modes = []
        if mode in {"all", "conda"}:
            self.modes.append("conda")
        if mode in {"all", "docker"}:
            self.modes.append("docker")
    
    def setUp(self):
        self.client = LearnwareClient()

    def _test_container_with_pip(self, mode):
        learnware_id = "00000147"
        learnware = self.client.load_learnware(learnware_id=learnware_id)
        with LearnwaresContainer(learnware, ignore_error=False, mode=mode) as env_container:
            learnware = env_container.get_learnwares_with_container()[0]
            input_array = np.random.random(size=(20, 23))
            print(learnware.predict(input_array))

    def _test_container_with_conda(self, mode):
        learnware_id = "00000148"
        learnware = self.client.load_learnware(learnware_id=learnware_id)
        with LearnwaresContainer(learnware, ignore_error=False, mode=mode) as env_container:
            learnware = env_container.get_learnwares_with_container()[0]
            input_array = np.random.random(size=(20, 204))
            print(learnware.predict(input_array))

    def test_container_with_pip(self):
        for mode in self.modes:
            self._test_container_with_pip(mode=mode)
    
    def test_container_with_conda(self):
        for mode in self.modes:
            self._test_container_with_conda(mode=mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="all", help="The mode to run container, must be in ['all', 'conda', 'docker']")
    args = parser.parse_args()

    assert args.mode in {"all", "conda", "docker"}, f"The mode must be in ['all', 'conda', 'docker'], instead of '{args.mode}'"
    runner = unittest.TextTestRunner()
    runner.run(parametrize(TestContainer, mode=args.mode))