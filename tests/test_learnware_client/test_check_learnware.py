import os
import unittest
import tempfile


from learnware.client import LearnwareClient


class TestCheckLearnware(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        email = "liujd@lamda.nju.edu.cn"
        token = "f7e647146a314c6e8b4e2e1079c4bca4"

        self.client = LearnwareClient()
        self.client.login(email, token)

    def test_check_learnware_pip(self):
        learnware_id = "00000154"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            LearnwareClient.check_learnware(self.zip_path)
    
    def test_check_learnware_conda(self):
        learnware_id = "00000148"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(learnware_id, self.zip_path)
            LearnwareClient.check_learnware(self.zip_path)


if __name__ == "__main__":
    unittest.main()
