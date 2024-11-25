from __future__ import annotations
import numpy as np

from .base import SystemStatSpecification
from ...tests.benchmarks import LLMBenchmark
from ...logger import get_module_logger

logger = get_module_logger("llm_base_vector_spec")


class LLMGeneralCapabilitySpecification(SystemStatSpecification):
    """Large Language Model Base Vector Specification"""

    def __init__(self):
        self.score_vector = None
        super(LLMGeneralCapabilitySpecification, self).__init__(type=self.__class__.__name__)

    def generate_stat_spec_from_system(self, model: TorchModel) -> np.ndarray:
        # model: foundation model
        # List[str]: each str is a dataset name
        dataset_names = LLMBenchmark().get_general_capability_datasets()

        pass

    def get_spec(self) -> np.ndarray:
        return self.score_vector