import traceback
from .base import BaseChecker
from ..learnware import Learnware
from ..client.container import LearnwaresContainer
from ..logger import get_module_logger

logger = get_module_logger("market_classes")


class CondaChecker(BaseChecker):
    def __init__(self, inner_checker, **kwargs):
        self.inner_checker = inner_checker
        super(CondaChecker, self).__init__(**kwargs)

    def __call__(self, learnware: Learnware) -> int:
        try:
            with LearnwaresContainer(learnware, ignore_error=False) as env_container:
                learnwares = env_container.get_learnwares_with_container()
                check_status, message = self.inner_checker(learnwares[0])
        except Exception as e:
            traceback.print_exc()
            message = f"Conda Checker failed due to installed learnware failed and {e}"
            logger.warning(message)
            message += "\n" + traceback.format_exc()
            return BaseChecker.INVALID_LEARNWARE, message
        return check_status, message
