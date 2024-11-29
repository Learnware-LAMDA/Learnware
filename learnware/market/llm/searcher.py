from typing import Optional

from .utils import is_llm
from ..base import BaseUserInfo, SearchResults
from ..easy import EasySearcher
from ..utils import parse_specification_type
from ...logger import get_module_logger


logger = get_module_logger("llm_searcher")


class LLMSearcher(EasySearcher):
    def search_learnware(self, user_info: BaseUserInfo) -> SearchResults:
        """Search helpful learnwares from learnware_list based on task vector specification

        Parameters
        ----------
        user_info : BaseUserInfo
            - user_info with semantic specifications and task vector specification

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            the first is the sorted list of task vector similarities
            the second is the sorted list of Learnware (single) by the task vector similarities
        """
        pass

    def __call__(
        self,
        user_info: BaseUserInfo,
        check_status: Optional[int] = None
    ) -> SearchResults:
        """Search learnwares based on user_info from learnwares with check_status.
           Employs LLM learnware search if specific requirements are met, otherwise resorts to homogeneous search methods.

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
        learnware_list = self.learnware_organizer.get_learnwares(check_status=check_status)
        semantic_search_result = self.semantic_searcher(learnware_list, user_info)

        learnware_list = [search_item.learnware for search_item in semantic_search_result.get_single_results()]
        if len(learnware_list) == 0:
            return SearchResults()

        if parse_specification_type(stat_specs=user_info.stat_info) is not None:
            if is_llm(stat_specs=user_info.stat_info, semantic_spec=user_info.semantic_spec):
                return self.search_learnware(user_info)
            return self.stat_searcher(learnware_list, user_info, max_search_num=1, search_method="greedy")
        else:
            return semantic_search_result