from __future__ import annotations
from typing import List

from .base import SystemStatSpecification
from ...tests.benchmarks import LearnwareBenchmarkManager, BenchmarkConfig
from ...logger import get_module_logger
from ...learnware import Learnware

logger = get_module_logger("llm_general_capability_spec")


class LLMGeneralCapabilitySpecification(SystemStatSpecification):
    """Large Language Model General Capability Specification"""

    def __init__(self):
        super(LLMGeneralCapabilitySpecification, self).__init__(type=self.__class__.__name__)

    def generate_stat_spec_from_system(self, learnware: Learnware, benchmark_configs: List[BenchmarkConfig]) -> dict:
        # model: foundation model
        dataset_names = LearnwareBenchmarkManager().list_benchmarks()

        pass

    def save(self, filepath: str):
        """Save the computed specification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path
        """
        raise NotImplementedError("save is not implemented")

    def load(self, filepath: str) -> bool:
        """Load a specification file in JSON format from the specified path.

        Parameters
        ----------
        filepath : str
            The specified loading path.

        Returns
        -------
        bool
            True if the specification is loaded successfully.
        """
        raise NotImplementedError("load is not implemented")
