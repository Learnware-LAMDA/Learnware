import os
import json
import zipfile
import unittest
import tempfile
import argparse

from learnware.client import LearnwareClient
from learnware.specification import generate_semantic_spec
from learnware.market import BaseUserInfo

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
        max_learnware_num = 2000
        semantic_spec = generate_semantic_spec()
        user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={})
        result = self.client.search_learnware(user_info, page_size=max_learnware_num)
        
        learnware_ids = result["single"]["learnware_ids"]
        keys = [key for key in result["single"]["semantic_specifications"][0]]
        print(f"result size: {len(learnware_ids)}")
        print(f"key in result: {keys}")

        failed_ids = []
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



def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllLearnware("test_all_learnware", email=None, token=None))
    return _suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
