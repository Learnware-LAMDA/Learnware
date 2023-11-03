from .base import BaseChecker, BaseOrganizer
from ..learnware import Learnware
from ..client.container import LearnwaresContainer


class CondaChecker(BaseChecker):
    def __init__(self, inner_checker, **kwargs):
        self.inner_checker = inner_checker
        super(CondaChecker, self).__init__(**kwargs)
    
    def __call__(self, learnware: Learnware) -> int:
    
        with LearnwaresContainer(learnware) as env_container:
            learnwares = env_container.get_learnwares_with_container()
            if len(learnwares) == 0:
                raise AssertionError(f"The env of learnware {learnware} installed failed")
            check_status = self.inner_checker(learnware[0])
        
        return check_status