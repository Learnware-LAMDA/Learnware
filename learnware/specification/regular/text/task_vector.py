from __future__ import annotations

import os
from typing import Any, Dict, List, Union

from ..base import RegularStatSpecification
from ....config import C
from ....logger import get_module_logger
from ....utils import allocate_cuda_idx, choose_device

logger = get_module_logger("RKMETextSpecification", "INFO")


class TaskVectorSpecification(RegularStatSpecification):
    """Task Vector Specification for Large Language Model"""

    def __init__(self, cuda_idx: int = None, **kwargs):
        """Initializing Task Vector Specification's parameters.
        
        Parameters
        ----------
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used. None indicates automatically choose device
        """
        self.task_vector = None
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self._device = choose_device(cuda_idx=self._cuda_idx)

        self.model_config = None

        super(TaskVectorSpecification, self).__init__(type=self.__class__.__name__)
    
    def _generate_models(self):
        """Initialize foundational model (e.g. RoBERTa) used for task vector generation.
        """
        pass
    
    def generate_stat_spec_from_data(
        self,
        X: List[str],
        verbose: bool = True,
        **kwargs
    ):
        pass

    def dist(self, VectorSpec2: TaskVectorSpecification) -> float:
        """Compute cosine similarity between two LLM task vectors.
        """
        pass