import json
import os
import tempfile
import unittest

from learnware.tests.benchmarks.config import LLMBenchmarkConfig
from learnware.specification import LLMGeneralCapabilitySpecification
from learnware.client import LearnwareClient

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TestGeneralCapabilitySpec(unittest.TestCase):
    @staticmethod
    def _test_general_spec(learnware, benchmark_configs):
        spec = LLMGeneralCapabilitySpecification()
        spec.generate_stat_spec_from_system(learnware=learnware, benchmark_configs=benchmark_configs)
        
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            spec_path = os.path.join(tempdir, "general_spec.json")
            spec.save(spec_path)

            with open(spec_path, "r") as f:
                data = json.load(f)
                assert data["type"] == "LLMGeneralCapabilitySpecification"

            spec2 = LLMGeneralCapabilitySpecification()
            spec2.load(spec_path)
            assert spec2.type == "LLMGeneralCapabilitySpecification"

    def test_general_spec(self):
        client = LearnwareClient()
        learnware = client.load_learnware(learnware_id="00002681")
        benchmark_configs = [
            LLMBenchmarkConfig(
                name="mmlu_anatomy",
                eval_metric="acc",
            )
        ]
        self._test_general_spec(learnware=learnware, benchmark_configs=benchmark_configs)


if __name__ == "__main__":
    unittest.main()