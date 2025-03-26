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
from learnware.specification.module import generate_generative_model_spec
from learnware.specification.regular.text import GenerativeModelSpecification

from text_generative_utils import DATASET, prepare_data

class TestGenerativeModelSpecification(unittest.TestCase):
    @staticmethod
    def _test_with_X(X):
        spec = GenerativeModelSpecification()
        spec.generate_stat_spec_from_data(X=X, dataset_text_field="txt")
        
        task_vector = spec.task_vector
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            spec_path = os.path.join(tempdir, "spec.pth")
            spec.save(spec_path)

            data = torch.load(spec_path, weights_only=True)
            assert data["type"] == "GenerativeModelSpecification"

            spec2 = GenerativeModelSpecification()
            spec2.load(spec_path)
            
            torch.testing.assert_close(task_vector.cpu(), spec2.task_vector.cpu())
            
            assert spec2.type == "GenerativeModelSpecification"
            
    @staticmethod
    def _test_with_dataset(dataset):
        spec = GenerativeModelSpecification()
        spec.generate_stat_spec_from_data(dataset=dataset)
        
        task_vector = spec.task_vector
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            spec_path = os.path.join(tempdir, "spec.pth")
            spec.save(spec_path)

            data = torch.load(spec_path, weights_only=True)
            assert data["type"] == "GenerativeModelSpecification"

            spec2 = GenerativeModelSpecification()
            spec2.load(spec_path)
            
            torch.testing.assert_close(task_vector.cpu(), spec2.task_vector.cpu())
            assert spec2.type == "GenerativeModelSpecification"
            
    @staticmethod
    def _test_with_generating_directly(X):
        spec = generate_generative_model_spec(X=X, dataset_text_field="name")
        
        task_vector = spec.task_vector
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            spec_path = os.path.join(tempdir, "spec.pth")
            spec.save(spec_path)

            data = torch.load(spec_path, weights_only=True)
            assert data["type"] == "GenerativeModelSpecification"

            spec2 = GenerativeModelSpecification()
            spec2.load(spec_path)
            
            torch.testing.assert_close(task_vector.cpu(), spec2.task_vector.cpu())
            assert spec2.type == "GenerativeModelSpecification"
            
    def test_loading_from_bwm(self):
        spec = GenerativeModelSpecification()
        _, model1 = spec._init_tokenizer_model(True)
        _, model2 = spec._init_tokenizer_model(False)
        
        params1, params2 = dict(model1.named_parameters()), dict(model2.named_parameters())
        for k in model1.state_dict():
            torch.testing.assert_close(params1[k].cpu(), params2[k].cpu())

    def test_generating_spec(self):
        train_dataset = prepare_data(DATASET["pubmedqa"])
        
        self._test_with_X(train_dataset["text"])
        self._test_with_dataset(train_dataset)
        self._test_with_dataset(train_dataset, beimingwu=False)
        
    def test_searching_spec(self):
        specs, learnwares = [], []
        for i, dataset_name in enumerate(["pubmedqa", "medmcqa"]):
            train_dataset = prepare_data(DATASET[dataset_name])
        
            spec = GenerativeModelSpecification()
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
