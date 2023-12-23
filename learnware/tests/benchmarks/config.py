from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BenchmarkConfig:
    name: str
    user_num: int
    learnware_ids: List[str]
    test_data_path: str
    train_data_path: Optional[str] = None
    extra_info_path: Optional[str] = None


benchmark_configs = {}
