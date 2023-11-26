import os
import json
import zipfile
import unittest
import tempfile

from learnware.client import LearnwareClient
from learnware.specification import Specification


class TestAllLearnware(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        dir_path = os.path.dirname(__file__)
        config_path = os.path.join(dir_path, "config.json")
        if not os.path.exists(config_path):
            data = {"email": None, "token": None}
            with open(config_path, "w") as file:
                json.dump(data, file)

        with open(config_path, "r") as file:
            data = json.load(file)
            email = data["email"]
            token = data["token"]

        if email is None or token is None:
            raise ValueError("Please set email and token in config.json.")
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
                with zipfile.ZipFile(zip_path, "r") as zip_file:
                    with zip_file.open("semantic_specification.json") as json_file:
                        semantic_spec = json.load(json_file)
                try:
                    LearnwareClient.check_learnware(zip_path, semantic_spec)
                    print(f"check learnware {idx} succeed")
                except:
                    failed_ids.append(idx)
                    print(f"check learnware {idx} failed!!!")

                print(f"The currently failed learnware ids: {failed_ids}")


if __name__ == "__main__":
    unittest.main()
