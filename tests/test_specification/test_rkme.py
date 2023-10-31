import os
import json
import string
import random
import unittest
import tempfile
import numpy as np

import learnware
import learnware.specification as specification
from learnware.specification import RKMEStatSpecification, RKMETextStatSpecification


class TestRKME(unittest.TestCase):
    def test_rkme(self):
        X = np.random.uniform(-10000, 10000, size=(5000, 200))
        rkme = specification.utils.generate_rkme_spec(X)

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            rkme_path = os.path.join(tempdir, "rkme.json")
            rkme.save(rkme_path)

            with open(rkme_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "RKMEStatSpecification"

            rkme2 = RKMEStatSpecification()
            rkme2.load(rkme_path)
            assert rkme2.type == "RKMEStatSpecification"
    
    def test_text_rkme(self):
        def generate_random_text_list(num, text_type="en", min_len=10, max_len=1000):
            text_list = []
            for i in range(num):
                length = random.randint(min_len, max_len)
                if text_type == "en":
                    characters = string.ascii_letters + string.digits + string.punctuation
                    result_str = "".join(random.choice(characters) for i in range(length))
                    text_list.append(result_str)
                elif text_type == "zh":
                    result_str = "".join(chr(random.randint(0x4e00, 0x9fff)) for i in range(length))
                    text_list.append(result_str)
                else:
                    raise ValueError("Type should be en or zh")
            return text_list
        
        def _test_text_rkme(X):
            rkme = specification.utils.generate_rkme_text_spec(X)

            with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
                rkme_path = os.path.join(tempdir, "rkme.json")
                rkme.save(rkme_path)

                with open(rkme_path, "r") as f:
                    data = json.load(f)
                    assert data["type"] == "RKMETextStatSpecification"

                rkme2 = RKMETextStatSpecification()
                rkme2.load(rkme_path)
                assert rkme2.type == "RKMETextStatSpecification"
                
                return rkme2.get_z().shape[1]

        dim1 = _test_text_rkme(generate_random_text_list(3000, "en"))
        dim2 = _test_text_rkme(generate_random_text_list(4000, "en"))
        dim3 = _test_text_rkme(generate_random_text_list(2000, "zh"))
        dim4 = _test_text_rkme(generate_random_text_list(5000, "zh"))
        
        assert dim1 == dim2 and dim2 == dim3 and dim3 == dim4


if __name__ == "__main__":
    unittest.main()
