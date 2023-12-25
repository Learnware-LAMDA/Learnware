import os
import json
import string
import random
import unittest
import tempfile

from learnware.specification import RKMETextSpecification
from learnware.specification import generate_stat_spec


class TestTextRKME(unittest.TestCase):
    @staticmethod
    def generate_random_text_list(num, text_type="en", min_len=10, max_len=1000):
            text_list = []
            for i in range(num):
                length = random.randint(min_len, max_len)
                if text_type == "en":
                    characters = string.ascii_letters + string.digits + string.punctuation
                    result_str = "".join(random.choice(characters) for i in range(length))
                    text_list.append(result_str)
                elif text_type == "zh":
                    result_str = "".join(chr(random.randint(0x4E00, 0x9FFF)) for i in range(length))
                    text_list.append(result_str)
                else:
                    raise ValueError("Type should be en or zh")
            return text_list

    @staticmethod
    def _test_text_rkme(X):
        rkme = generate_stat_spec(type="text", X=X)

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            rkme_path = os.path.join(tempdir, "rkme.json")
            rkme.save(rkme_path)

            with open(rkme_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "RKMETextSpecification"

            rkme2 = RKMETextSpecification()
            rkme2.load(rkme_path)
            assert rkme2.type == "RKMETextSpecification"

            return rkme2.get_z().shape[1]

    def test_text_rkme(self):
        dim1 = self._test_text_rkme(self.generate_random_text_list(3000, "en"))
        dim2 = self._test_text_rkme(self.generate_random_text_list(100, "en"))
        dim3 = self._test_text_rkme(self.generate_random_text_list(50, "zh"))
        dim4 = self._test_text_rkme(self.generate_random_text_list(5000, "zh"))
        dim5 = self._test_text_rkme(self.generate_random_text_list(1, "zh"))

        assert dim1 == dim2 and dim2 == dim3 and dim3 == dim4 and dim4 == dim5


if __name__ == "__main__":
    unittest.main()
