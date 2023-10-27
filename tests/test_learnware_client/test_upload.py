import os
import unittest
import tempfile

from learnware.client import LearnwareClient


class TestAllLearnware(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUpClass()
        email = "liujd@lamda.nju.edu.cn"
        token = "f7e647146a314c6e8b4e2e1079c4bca4"

        self.client = LearnwareClient()
        self.client.login(email, token)

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
        semantic_spec = self.client.create_semantic_specification(
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
    unittest.main()
