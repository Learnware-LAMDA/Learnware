import os
import json
import zipfile
import unittest
import tempfile

from learnware.client import LearnwareClient


class TestCheckLearnware(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        self.client = LearnwareClient()

    def test_check_learnware_pip_only_zip(self):
        learnware_id = "00000208"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            LearnwareClient.check_learnware(self.zip_path)

    def test_check_learnware_pip(self):
        learnware_id = "00000208"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            semantic_spec = self.client.get_semantic_specification(learnware_id)
            LearnwareClient.check_learnware(self.zip_path, semantic_spec)

    def test_check_learnware_conda(self):
        learnware_id = "00000148"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            semantic_spec = self.client.get_semantic_specification(learnware_id)
            LearnwareClient.check_learnware(self.zip_path, semantic_spec)

    def test_check_learnware_dependency(self):
        learnware_id = "00000147"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            semantic_spec = self.client.get_semantic_specification(learnware_id)
            LearnwareClient.check_learnware(self.zip_path, semantic_spec)

    def test_check_learnware_image(self):
        learnware_id = "00000677"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            semantic_spec = self.client.get_semantic_specification(learnware_id)
            LearnwareClient.check_learnware(self.zip_path, semantic_spec)

    def test_check_learnware_text(self):
        learnware_id = "00000662"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            semantic_spec = self.client.get_semantic_specification(learnware_id)
            LearnwareClient.check_learnware(self.zip_path, semantic_spec)


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestCheckLearnware("test_check_learnware_pip_only_zip"))
    _suite.addTest(TestCheckLearnware("test_check_learnware_pip"))
    _suite.addTest(TestCheckLearnware("test_check_learnware_conda"))
    _suite.addTest(TestCheckLearnware("test_check_learnware_dependency"))
    _suite.addTest(TestCheckLearnware("test_check_learnware_image"))
    _suite.addTest(TestCheckLearnware("test_check_learnware_text"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
