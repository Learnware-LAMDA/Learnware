import unittest

import numpy as np

from learnware.client import LearnwareClient
from learnware.client.container import LearnwaresContainer


class TestContainer(unittest.TestCase):
    def __init__(self, method_name="runTest", mode="all"):
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


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestContainer("test_container_with_pip", mode="all"))
    _suite.addTest(TestContainer("test_container_with_conda", mode="all"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
