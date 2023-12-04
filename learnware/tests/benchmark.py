from dataclasses import dataclass, field
from typing import Optional, List
from ..learnware import Learnware
@dataclass
class BenchmarkConfig:
    datasert_url: str
    learnware_ids: List[str]
    userdata_url: Optional[str] = None
    
    
class LearnwareBenchmark:
    pass
