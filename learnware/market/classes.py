from .base import BaseChecker, BaseOrganizer
from ..learnware import Learnware
from ..client.container import LearnwaresContainer
from ..logger import get_module_logger

logger = get_module_logger('market_classes')

class CondaChecker(BaseChecker):
    def __init__(self, inner_checker, **kwargs):
        self.inner_checker = inner_checker
        super(CondaChecker, self).__init__(**kwargs)

    def __call__(self, learnware: Learnware) -> int:
        with LearnwaresContainer(learnware) as env_container:
            if not all(env_container.get_learnware_flags()):
                logger.warning(f"The env of learnware {learnware} installed failed")
                return BaseChecker.INVALID_LEARNWARE
            learnwares = env_container.get_learnwares_with_container()
            check_status = self.inner_checker(learnwares[0])

        return check_status
