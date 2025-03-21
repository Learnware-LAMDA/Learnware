from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Union


@dataclass
class BenchmarkConfig:
    name: str
    user_num: int
    learnware_ids: List[str]
    test_data_path: str
    train_data_path: Optional[str] = None
    extra_info_path: Optional[str] = None


@dataclass
class LLMBenchmarkConfig:
    name: str
    # HF dataset options
    dataset_path: Optional[str] = None
    subset_name: Optional[str] = None
    dataset_kwargs: Optional[dict] = None
    train_split: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: Optional[str] = None
    # evaluation options
    eval_metric: Optional[str] = None
    score_function: Optional[Callable] = None
    # formatting / prompting options
    preprocess_function: Optional[Callable] = None
    response_template: Optional[str] = None


benchmark_configs: Dict[str, Union[BenchmarkConfig, LLMBenchmarkConfig]] = {}

llm_general_capability_benchmark_configs: Dict[str, LLMBenchmarkConfig] = {}
