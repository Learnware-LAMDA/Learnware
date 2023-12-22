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


benchmark_configs = {
    "example": BenchmarkConfig(
        name="example",
        user_num=3,
        learnware_ids=["00001951", "00001980", "00001987"],
        test_data_path="example_path1",
        train_data_path="example_path2",
        extra_info_path="example_path3",
    )
}
