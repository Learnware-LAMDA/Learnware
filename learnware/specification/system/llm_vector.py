from __future__ import annotations
import numpy as np

from .base import SystemStatSpecification
from ...model import TorchModel
from ...tests.benchmarks import LearnwareBenchmarkManager
from ...logger import get_module_logger

logger = get_module_logger("llm_general_capability_spec")


class LLMGeneralCapabilitySpecification(SystemStatSpecification):
    """Large Language Model General Capability Specification"""

    def __init__(self):
        self.score_vector = None
        super(LLMGeneralCapabilitySpecification, self).__init__(type=self.__class__.__name__)

    def generate_stat_spec_from_system(self, model: TorchModel) -> np.ndarray:
        # model: foundation model
        dataset_names = LearnwareBenchmarkManager().list_benchmarks()

        pass

    def get_spec(self) -> np.ndarray:
        return self.score_vector
