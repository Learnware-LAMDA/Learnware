import os
import json
import torch
import unittest
import tempfile
import numpy as np

import learnware
from learnware.specification import RKMEStatSpecification, RKMEImageStatSpecification
from learnware.specification import generate_rkme_image_spec, generate_rkme_spec


class TestRKME(unittest.TestCase):
    def test_rkme(self):
        X = np.random.uniform(-10000, 10000, size=(5000, 200))
        rkme = generate_rkme_spec(X)
        rkme.generate_stat_spec_from_data(X)

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            rkme_path = os.path.join(tempdir, "rkme.json")
            rkme.save(rkme_path)

            with open(rkme_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "RKMEStatSpecification"

            rkme2 = RKMEStatSpecification()
            rkme2.load(rkme_path)
            assert rkme2.type == "RKMEStatSpecification"

    def test_image_rkme(self):
        def _test_image_rkme(X):
            image_rkme = generate_rkme_image_spec(X, steps=10)

            with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
                rkme_path = os.path.join(tempdir, "rkme.json")
                image_rkme.save(rkme_path)

                with open(rkme_path, "r") as f:
                    data = json.load(f)
                    assert data["type"] == "RKMEImageStatSpecification"

                rkme2 = RKMEImageStatSpecification()
                rkme2.load(rkme_path)
                assert rkme2.type == "RKMEImageStatSpecification"

        _test_image_rkme(np.random.randint(0, 255, size=(2000, 3, 32, 32)))
        _test_image_rkme(np.random.randint(0, 255, size=(2000, 3, 128, 128)))
        _test_image_rkme(np.random.randint(0, 255, size=(2000, 3, 128, 128)) / 255)

        _test_image_rkme(torch.randint(0, 255, (2000, 3, 32, 32)))
        _test_image_rkme(torch.randint(0, 255, (2000, 3, 128, 128)))
        _test_image_rkme(torch.randint(0, 255, (2000, 3, 128, 128)) / 255)


if __name__ == "__main__":
    unittest.main()
