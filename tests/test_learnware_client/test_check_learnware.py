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
        self.learnware_id = "00000154"

    def test_check_learnware(self):
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            self.zip_path = os.path.join(tempdir, "test.zip")
            self.client.download_learnware(self.learnware_id, self.zip_path)
            LearnwareClient.check_learnware(self.zip_path)


if __name__ == "__main__":
    unittest.main()