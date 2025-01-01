import json
import os
import tempfile
import unittest

import numpy as np
import torch

import sys

from learnware.learnware.base import Learnware
from learnware.market.llm import LLMStatSearcher
from learnware.specification.base import Specification
from learnware.specification.regular.text import GenerativeModelSpecification


# Import from our project

if os.path.expanduser(os.environ["LIB_PATH"]) not in sys.path:
    sys.path.append(os.path.expanduser(os.environ["LIB_PATH"]))


from src.datasets.llm.utils import set_seed, prepare_train_data
from src.datasets.llm.benchmark import Benchmark


class TestGenerativeModelSpecification(unittest.TestCase):
    @staticmethod
    def _test_with_X(X):
        spec = GenerativeModelSpecification(max_steps=5)
        spec.generate_stat_spec_from_data(X=X, dataset_text_field="txt")
        
        task_vector = spec.task_vector
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            spec_path = os.path.join(tempdir, "spec.pth")
            spec.save(spec_path)

            data = torch.load(spec_path, weights_only=True)
            assert data["type"] == "GenerativeModelSpecification"

            spec2 = GenerativeModelSpecification()
            spec2.load(spec_path)
            
            torch.testing.assert_close(task_vector, spec2.task_vector)
            
            assert spec2.type == "GenerativeModelSpecification"
            
    @staticmethod
    def _test_with_dataset(dataset):
        spec = GenerativeModelSpecification(max_steps=5)
        spec.generate_stat_spec_from_data(dataset=dataset)
        
        task_vector = spec.task_vector
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            spec_path = os.path.join(tempdir, "spec.pth")
            spec.save(spec_path)

            data = torch.load(spec_path, weights_only=True)
            assert data["type"] == "GenerativeModelSpecification"

            spec2 = GenerativeModelSpecification()
            spec2.load(spec_path)
            
            torch.testing.assert_close(task_vector, spec2.task_vector)
            assert spec2.type == "GenerativeModelSpecification"

    # def test_generating_spec(self):
    #     benchmark = Benchmark("medical")
    #     train_dataset = benchmark.get_user_dataset("pubmedqa")
        
    #     self._test_with_X(train_dataset["text"])
    #     self._test_with_dataset(train_dataset)
        
    def test_searching_spec(self):
        benchmark = Benchmark("medical")
        
        specs, learnwares = [], []
        for i, dataset_name in enumerate(["pubmedqa", "medmcqa"]):
            train_dataset = benchmark.get_user_dataset("pubmedqa")
        
            spec = GenerativeModelSpecification(max_steps=5)
            spec.generate_stat_spec_from_data(dataset=train_dataset)
        
            specs.append(spec)
            learnwares.append(Learnware(str(i), None, Specification(
                stat_spec={spec.type: spec}
                ), ""))
        
        searcher = LLMStatSearcher(None)
        searcher._search_by_taskvector_spec_single(
            learnwares,
            specs[-1],
            specs[-1].type
        )


if __name__ == "__main__":
    unittest.main()
