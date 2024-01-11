import os
import json
import unittest
import tempfile
import numpy as np

from learnware.specification import RKMETableSpecification, RKMEImageSpecification, RKMETextSpecification
from learnware.specification import generate_stat_spec


class TestTableRKME(unittest.TestCase):
    @staticmethod
    def _test_table_rkme(X):
        rkme = generate_stat_spec(type="table", X=X)

        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            rkme_path = os.path.join(tempdir, "rkme.json")
            rkme.save(rkme_path)

            with open(rkme_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "RKMETableSpecification"

            rkme2 = RKMETableSpecification()
            rkme2.load(rkme_path)
            assert rkme2.type == "RKMETableSpecification"

    def test_table_rkme(self):
        self._test_table_rkme(np.random.uniform(-10000, 10000, size=(5000, 200)))
        self._test_table_rkme(np.random.uniform(-10000, 10000, size=(10000, 100)))
        self._test_table_rkme(np.random.uniform(-10000, 10000, size=(5, 20)))
        self._test_table_rkme(np.random.uniform(-10000, 10000, size=(1, 50)))
        self._test_table_rkme(np.random.uniform(-10000, 10000, size=(100, 150)))


if __name__ == "__main__":
    unittest.main()
