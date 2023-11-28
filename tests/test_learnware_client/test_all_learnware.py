import os
import json
import zipfile
import unittest
import tempfile
import argparse

from learnware.client import LearnwareClient
from learnware.specification import generate_semantic_spec
from learnware.market import BaseUserInfo
from learnware.tests import parametrize

class TestAllLearnware(unittest.TestCase):
    client = LearnwareClient()
    
    def __init__(self, method_name='runTest', email=None, token=None):
        super(TestAllLearnware, self).__init__(method_name)
        self.email = email
        self.token = token
        
        if self.email is not None and self.token is not None:
            self.client.login(self.email, self.token)
        else:
            print("Client doest not login, all tests will be ignored!")

    @unittest.skipIf(not client.is_login(), "Client doest not login!")
    def test_all_learnware(self):
        max_learnware_num = 1000
        semantic_spec = generate_semantic_spec()
        user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={})
        result = self.client.search_learnware(user_info, page_size=max_learnware_num)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, required=False, help="The email to login learnware client")
    parser.add_argument("--token", type=str, required=False, help="The token to login learnware client")
    args = parser.parse_args()

    runner = unittest.TextTestRunner()
    runner.run(parametrize(TestAllLearnware, email=args.email, token=args.token))