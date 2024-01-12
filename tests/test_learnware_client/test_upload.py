import os
import json
import unittest
import tempfile

from learnware.client import LearnwareClient
from learnware.specification import generate_semantic_spec


class TestUpload(unittest.TestCase):
    client = LearnwareClient()

    @classmethod
    def setUpClass(cls) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")

        if not os.path.exists(config_path):
            data = {"email": None, "token": None}
            with open(config_path, "w") as file:
                json.dump(data, file)

        with open(config_path, "r") as file:
            data = json.load(file)
            email = data.get("email")
            token = data.get("token")

        if email is None or token is None:
            print("Please set email and token in config.json.")
        else:
            cls.client.login(email, token)

    def _skip_test(self):
        if not self.client.is_login():
            print("Client does not login!")
            return True
        return False

    def test_upload(self):
        if not self._skip_test():
            input_description = {
                "Dimension": 13,
                "Description": {"0": "age", "1": "weight", "2": "body length", "3": "animal type", "4": "claw length"},
            }
            output_description = {
                "Dimension": 2,
                "Description": {"0": "cat", "1": "not cat"},
            }
            semantic_spec = generate_semantic_spec(
                name="learnware_example",
                description="Just a example for uploading a learnware",
                data_type="Table",
                task_type="Classification",
                library_type="Scikit-learn",
                scenarios=["Business", "Financial"],
                license="MIT",
                input_description=input_description,
                output_description=output_description,
            )
            assert isinstance(semantic_spec, dict)

            download_learnware_id = "00000084"
            with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
                zip_path = os.path.join(tempdir, "test.zip")
                self.client.download_learnware(download_learnware_id, zip_path)
                learnware_id = self.client.upload_learnware(
                    learnware_zip_path=zip_path, semantic_specification=semantic_spec
                )

                uploaded_ids = [learnware["learnware_id"] for learnware in self.client.list_learnware()]
                assert learnware_id in uploaded_ids

                self.client.delete_learnware(learnware_id)
                uploaded_ids = [learnware["learnware_id"] for learnware in self.client.list_learnware()]
                assert learnware_id not in uploaded_ids


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestUpload("test_upload"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
