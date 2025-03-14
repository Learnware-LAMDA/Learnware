from __future__ import annotations
import traceback
from typing import List, Dict, Optional
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMGeneralCapabilitySpecification(SystemStatSpecification):
    """Large Language Model General Capability Specification"""

    benchmark_configs: List[LLMBenchmarkConfig] = general_capability_benchmark_configs

    def __init__(self):
        self.score_dict = None
        super(LLMGeneralCapabilitySpecification, self).__init__(type=self.__class__.__name__)

    @staticmethod
    def _get_scores(learnware: Learnware, benchmark_configs: List[LLMBenchmarkConfig]) -> Dict:
        """Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework to evaluate learnware according to benchmark_configs and compute score dict.

        Parameters
        ----------
        learnware : Learnware
            Learnware to generate General Capability Specification.
        benchmark_configs : Optional[List[LLMBenchmarkConfig]]
            List of LLMBenchmarkConfig.
        
        Returns
        -------
        Dict[LLMBenchmarkConfig, float]
            Scores of all benchmark_configs.
        """
        learnware.instantiate_model()
        base_model = learnware.get_model().get_model()
        task_manager = lm_eval.tasks.TaskManager()

        score_dict = {}
        for config in benchmark_configs:
            try:
                lm_obj = HFLM(pretrained=base_model, batch_size="auto")
                results = lm_eval.simple_evaluate(
                    model=lm_obj,
                    tasks=[config.name],
                    task_manager=task_manager,
                )
                
                if config.score_function:
                    score = config.score_function(results)
                else:
                    score = results['results'][config.name][f'{config.eval_metric},none'] * 100
                    score = round(score, 2)
                logger.info(f"Name: {config.name}, Score: {score}")
                score_dict[config.name] = score
            
            except Exception as e:
                traceback.print_exc()
                message = f"Evaluation of {config.name} failed! Due to {repr(e)}."
                logger.warning(message)
            
        return score_dict

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
                if config.eval_metric == None and config.score_function == None:
                    raise Exception("Must specify an evaluation metric or a score computing function in a LLMBenchmarkConfig object to get the evaluation score.")
        else:
            logger.info("No passed benchmark_configs. Set benchmark_configs by default.")
            benchmark_configs = self.benchmark_configs 
        if update_existing:
            logger.info("Update existing LLMGeneralCapabilitySpecification.")
            self.score_dict = self._get_scores(learnware, benchmark_configs)
        else:
            existing_config_names = []
            self.score_dict = {}
            general_spec = learnware.get_specification().get_stat_spec_by_name("LLMGeneralCapabilitySpecification")
            if general_spec:
                existing_config_names = list(general_spec.score_dict.keys())
                self.score_dict = general_spec.score_dict.copy()
                logger.info("LLMGeneralCapabilitySpecification exists in learnware. Try to update...")
                for k, v in general_spec.score_dict.items():
                    logger.info(f"Existing scores: Name: {k}, Score: {v}")
            new_configs = [config for config in benchmark_configs if config.name not in existing_config_names]
            if new_configs:
                new_score_dict = self._get_scores(learnware, new_configs)
                self.score_dict.update(new_score_dict)
            else:
                logger.info("All LLMBenchmarkConfig have been evaluated before. No update.")


    def __str__(self):
        spec_to_save = self.get_states()
        return json.dumps(spec_to_save, separators=(",", ":"))
    
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
