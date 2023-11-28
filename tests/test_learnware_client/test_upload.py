import os
import argparse
import unittest
import tempfile

from learnware.client import LearnwareClient
from learnware.specification import generate_semantic_spec
from learnware.tests import parametrize

class TestUpload(unittest.TestCase):
    client = LearnwareClient()
    
    def __init__(self, method_name='runTest', email=None, token=None):
        super(TestUpload, self).__init__(method_name)
        self.email = email
        self.token = token
        
        if self.email is not None and self.token is not None:
            self.client.login(self.email, self.token)
        else:
            print("Client doest not login, all tests will be ignored!")

    @unittest.skipIf(not client.is_login(), "Client doest not login!")
    def test_upload(self):
        input_description = {
            "Dimension": 13,
            "Description": {"0": "age", "1": "weight", "2": "body length", "3": "animal type", "4": "claw length"},
        }
        output_description = {
            "Dimension": 1,
            "Description": {
                "0": "the probability of being a cat",
            },
        }
        semantic_spec = generate_semantic_spec(
            name="learnware_example",
            description="Just a example for uploading a learnware",
            data_type="Table",
            task_type="Classification",
            library_type="Scikit-learn",
            scenarios=["Business", "Financial"],
            input_description=input_description,
            output_description=output_description,
        )
        assert isinstance(semantic_spec, dict)

        download_learnware_id = "00000084"
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            zip_path = os.path.join(tempdir, f"test.zip")
            self.client.download_learnware(download_learnware_id, zip_path)
            learnware_id = self.client.upload_learnware(
                learnware_zip_path=zip_path, semantic_specification=semantic_spec
            )

            uploaded_ids = [learnware["learnware_id"] for learnware in self.client.list_learnware()]
            assert learnware_id in uploaded_ids

            self.client.delete_learnware(learnware_id)
            uploaded_ids = [learnware["learnware_id"] for learnware in self.client.list_learnware()]
            assert learnware_id not in uploaded_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, required=False, help="The email to login learnware client")
    parser.add_argument("--token", type=str, required=False, help="The token to login learnware client")
    args = parser.parse_args()

    runner = unittest.TextTestRunner()
    runner.run(parametrize(TestUpload, email=args.email, token=args.token))