from dataclasses import dataclass, field
from typing import Optional, List
from ...learnware import Learnware

@dataclass
class OnlineBenchmark:
    learnware_ids: List[str]
    user_num: int
    unlabeled_data_path: str
    labeled_data_path: Optional[str] = None
    extra_info_path: Optional[str] = None

online_benchmarks = {
    "example": OnlineBenchmark(
        learnware_ids=["00001951", "00001980", "00001987"],
        user_num=3,
        unlabeled_data_path="example_path1",
        labeled_data_path="example_path2",
        extra_info_path="example_path3"
    )
}