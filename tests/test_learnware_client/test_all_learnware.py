import os
import unittest
import tempfile

from learnware.client import LearnwareClient
from learnware.specification import Specification


class TestAllLearnware(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        email = "liujd@lamda.nju.edu.cn"
        token = "f7e647146a314c6e8b4e2e1079c4bca4"

        self.client = LearnwareClient()
        self.client.login(email, token)

    def test_all_learnware(self):
        max_learnware_num = 1000
        semantic_spec = dict()
        semantic_spec["Data"] = {"Type": "Class", "Values": []}
        semantic_spec["Task"] = {"Type": "Class", "Values": []}
        semantic_spec["Library"] = {"Type": "Class", "Values": []}
        semantic_spec["Scenario"] = {"Type": "Tag", "Values": []}
        semantic_spec["Name"] = {"Type": "String", "Values": ""}
        semantic_spec["Description"] = {"Type": "String", "Values": ""}

        specification = Specification(semantic_spec=semantic_spec)
        result = self.client.search_learnware(specification, page_size=max_learnware_num)
        print(f"result size: {len(result)}")
        print(f"key in result: {[key for key in result[0]]}")

        failed_ids = []
        learnware_ids = [res["learnware_id"] for res in result]
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            for idx in learnware_ids:
                zip_path = os.path.join(tempdir, f"test_{idx}.zip")
                self.client.download_learnware(idx, zip_path)
                try:
                    LearnwareClient.check_learnware(zip_path)
                    print(f"check learnware {idx} succeed")
                except:
                    failed_ids.append(idx)
                    print(f"check learnware {idx} failed!!!")

        print(f"failed learnware ids: {failed_ids}")


if __name__ == "__main__":
    unittest.main()
