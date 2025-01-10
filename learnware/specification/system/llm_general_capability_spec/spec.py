from __future__ import annotations
from typing import List, Optional
import lm_eval
from lm_eval.models.huggingface import HFLM 
import codecs
import json
import os

from .config import general_capability_benchmark_configs
from ..base import SystemStatSpecification
from ....tests.benchmarks import LLMBenchmarkConfig
from ....logger import get_module_logger

logger = get_module_logger("llm_general_capability_spec")


class LLMGeneralCapabilitySpecification(SystemStatSpecification):
    """Large Language Model General Capability Specification"""

    benchmark_configs: List[LLMBenchmarkConfig] = general_capability_benchmark_configs

    def __init__(self):
        self.score_dict = None
        super(LLMGeneralCapabilitySpecification, self).__init__(type=self.__class__.__name__)

    @staticmethod
    def _evaluate(learnware: Learnware, benchmark_configs: List[LLMBenchmarkConfig]):
        """Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework to evaluate learnware according to benchmark_configs.

        Parameters
        ----------
        learnware : Learnware
            Learnware to generate General Capability Specification.
        benchmark_configs : Optional[List[LLMBenchmarkConfig]]
            List of LLMBenchmarkConfig, set to self.benchmark_configs if None.
        """
        learnware.instantiate_model()
        base_model = learnware.get_model().get_model()
        task_list = [config.name for config in benchmark_configs]
        
        lm_obj = HFLM(pretrained=base_model, batch_size=16)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=task_list,
            task_manager=task_manager,
        )
        return results

    def generate_stat_spec_from_system(
        self,
        learnware: Learnware,
        benchmark_configs: Optional[List[LLMBenchmarkConfig]] = None,
        update_existing: bool = False,
    ):
        """Construct Large Language Model General Capability Specification for Learnware.

        Parameters
        ----------
        learnware : Learnware
            Learnware to generate General Capability Specification.
        benchmark_configs : Optional[List[LLMBenchmarkConfig]]
            List of LLMBenchmarkConfig, set to self.benchmark_configs if None.
        update_existing : bool
            A flag indicating whether to update existing General Capability Specification's scores dict, by default false.
        """
        if benchmark_configs:
            for config in benchmark_configs:
                if config.eval_metric == None:
                    raise Exception("Must specify a evaluation metric in a LLMBenchmarkConfig object to evaluate learnware on it.")
        else:
            benchmark_configs = self.benchmark_configs 
        self.score_dict = {}
        if update_existing:
            results = self._evaluate(learnware, benchmark_configs)
            for config in benchmark_configs:
                self.score_dict[config.name] = results['results'][config.name][f'{config.eval_metric},none']
        else:
            exist_config_list = []
            general_spec = learnware.get_specification().get_stat_spec_by_name("LLMGeneralCapabilitySpecification")
            if general_spec:
                exist_config_list = list(general_spec.score_dict.keys())
                self.score_dict = general_spec.score_dict.copy()
            remain_config_list = [config.name for config in benchmark_configs if config.name not in exist_config_list]
            if remain_config_list:
                results = self._evaluate(learnware, remain_config_list)
                for config in remain_config_list:
                    self.score_dict[config.name] = results['results'][config.name][f'{config.eval_metric},none']

    def save(self, filepath: str):
        """Save the computed specification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path
        """
        save_path = filepath
        spec_to_save = self.get_states()
        with codecs.open(save_path, "w", encoding="utf-8") as fout:
            json.dump(spec_to_save, fout, separators=(",", ":"))

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
        load_path = filepath
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            spec_load = json.loads(obj_text)

            for d in self.get_states():
                if d in spec_load.keys():
                    if d == "type" and spec_load[d] != self.type:
                        raise TypeError(
                            f"The type of loaded Specification ({spec_load[d]}) is different from the expected type ({self.type})!"
                        )
                    setattr(self, d, spec_load[d])
