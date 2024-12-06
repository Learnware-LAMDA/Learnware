from typing import Optional

from ..base import BaseUserInfo, SearchResults, BasicSearcher
from ...logger import get_module_logger

logger = get_module_logger("llm_searcher")


class LLMStatSearcher(BasicSearcher):
    SPEC_TYPES = ["TaskVectorSpecification"]

    def is_applicable_user(self, user_info: BaseUserInfo, verbose: bool = True) -> bool:
        stat_specs = user_info.stat_info
        semantic_spec = user_info.semantic_spec
        try:
            if "TaskVectorSpecification" not in stat_specs:
                if verbose:
                    logger.warning("TaskVectorSpecification is not provided in stat_info.")
                return False

            semantic_data_type = semantic_spec["Data"]["Values"]
            if len(semantic_data_type) > 0 and semantic_data_type != ["Text"]:
                logger.warning("User doesn't provide correct data type, it must be Text.")
                return False

            semantic_task_type = semantic_spec["Task"]["Values"]
            if len(semantic_task_type) > 0 and semantic_task_type != ["Text Generation"]:
                logger.warning("User doesn't provide correct task type, it must be Text Generation.")
                return False

            return True
        except Exception as err:
            if verbose:
                logger.warning("Invalid llm search information provided.")
            return False

    def __call__(
        self,
        user_info: BaseUserInfo,
        check_status: Optional[int] = None,
        max_search_num: int = 5,
        search_method: str = "greedy",
    ) -> SearchResults:
        """Employ LLM learnware search based on user_info from learnwares with check_status.

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        check_status : int, optional
            - None: search from all learnwares
            - Others: search from learnwares with check_status

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            the first is the sorted list of rkme dist
            the second is the sorted list of Learnware (single) by the rkme dist
        """
        pass
