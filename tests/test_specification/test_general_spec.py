import json
import os
import tempfile
import unittest

from learnware.specification.system.llm_general_capability_spec.config import test_benchmark_configs
from learnware.specification import LLMGeneralCapabilitySpecification
from learnware.client import LearnwareClient
from learnware.market import instantiate_learnware_market
from learnware.specification import generate_semantic_spec
from learnware.market import LearnwareMarket

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        self._test_general_spec(learnware, test_benchmark_configs)
    
    @staticmethod
    def _prepare_learnware_market() -> LearnwareMarket:
        """initialize learnware market"""
        llm_market = instantiate_learnware_market(market_id="llm_test", name="llm", rebuild=True)
        semantic_spec = generate_semantic_spec(
            name="Qwen/Qwen2.5-0.5B",
            description="Qwen/Qwen2.5-0.5B",
            data_type="Text",
            model_type="Base Model",
            task_type="Text Generation",
            library_type="PyTorch",
            scenarios=["Others"],
            license="Others",
            input_description=None,
            output_description=None,
        )
        client = LearnwareClient()
        with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
            zip_path = os.path.join(tempdir, "learnware.zip")
            client.download_learnware(learnware_id="00002681", save_path=zip_path)
            llm_market.add_learnware(zip_path, semantic_spec)
        return llm_market

    def test_in_checker_organizer(self):
        llm_market = self._prepare_learnware_market()
        learnware_ids = llm_market.get_learnware_ids()
        llm_market.learnware_organizer._update_learnware_general_capability_spec(learnware_ids)


if __name__ == "__main__":
    unittest.main()