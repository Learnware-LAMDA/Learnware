import os
import json
import torch
import unittest
import tempfile
import numpy as np

from learnware.specification import RKMEImageSpecification
from learnware.specification import generate_stat_spec


class TestImageRKME(unittest.TestCase):
    @staticmethod
    def _test_image_rkme(X):
        image_rkme = generate_stat_spec(type="image", X=X, steps=10)

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            rkme_path = os.path.join(tempdir, "rkme.json")
            image_rkme.save(rkme_path)

            with open(rkme_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "RKMEImageSpecification"

            rkme2 = RKMEImageSpecification()
            rkme2.load(rkme_path)
            assert rkme2.type == "RKMEImageSpecification"
                
    def test_image_rkme(self):
        self._test_image_rkme(np.random.randint(0, 255, size=(2000, 3, 32, 32)))
        self._test_image_rkme(np.random.randint(0, 255, size=(100, 1, 128, 128)))
        self._test_image_rkme(np.random.randint(0, 255, size=(50, 3, 128, 128)) / 255)
        self._test_image_rkme(torch.randint(0, 255, (2000, 3, 32, 32)))
        self._test_image_rkme(torch.randint(0, 255, (20, 3, 128, 128)))
        self._test_image_rkme(torch.randint(0, 255, (1, 1, 128, 128)) / 255)

if __name__ == "__main__":
    unittest.main()
