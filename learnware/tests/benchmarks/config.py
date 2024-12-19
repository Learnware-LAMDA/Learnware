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
    preprocess_function: Optional[Callable] = None


benchmark_configs: Dict[str, Union[BenchmarkConfig, LLMBenchmarkConfig]] = {}

llm_general_capability_benchmark_configs: Dict[str, LLMBenchmarkConfig] = {}
