import os
import json
import string
import random
import torch
import unittest
import tempfile
import numpy as np

from learnware.specification import RKMETableSpecification, HeteroMapTableSpecification
from learnware.specification import generate_stat_spec
from learnware.market.heterogeneous.organizer import HeteroMap

class TestTableRKME(unittest.TestCase):
    
    def setUp(self):
        self.hetero_map = HeteroMap()
        
    def _test_hetero_spec(self, X):
        rkme: RKMETableSpecification = generate_stat_spec(type="table", X=X)
        hetero_spec = self.hetero_map.hetero_mapping(rkme_spec=rkme, features=dict())
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            rkme_path = os.path.join(tempdir, "rkme.json")
            hetero_spec.save(rkme_path)

            with open(rkme_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "HeteroMapTableSpecification"

            rkme2 = HeteroMapTableSpecification()
            rkme2.load(rkme_path)
            assert rkme2.type == "HeteroMapTableSpecification"
        
            
    def test_hetero_rkme(self):
        self._test_hetero_spec(np.random.uniform(-10000, 10000, size=(5000, 200)))
        self._test_hetero_spec(np.random.uniform(-10000, 10000, size=(10000, 100)))
        self._test_hetero_spec(np.random.uniform(-10000, 10000, size=(5, 20)))
        self._test_hetero_spec(np.random.uniform(-10000, 10000, size=(1, 50)))
        self._test_hetero_spec(np.random.uniform(-10000, 10000, size=(100, 150)))
        
if __name__ == "__main__":
    unittest.main()
