import os
import json
import unittest
import tempfile
import numpy as np

import learnware
import learnware.specification as specification
from learnware.specification import RKMEStatSpecification


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


if __name__ == "__main__":
    unittest.main()
