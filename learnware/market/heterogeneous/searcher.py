from typing import List

from .utils import is_hetero
from ..base import BaseUserInfo, SearchResults
from ..easy import EasyStatSearcher
from ...learnware import Learnware
from ...logger import get_module_logger

logger = get_module_logger("hetero_searcher")


class HeteroStatSearcher(EasyStatSearcher):
    SPEC_TYPES = ["HeteroMapTableSpecification"]

    def is_applicable_learnware(self, learnware: Learnware) -> bool:
        if not super(HeteroStatSearcher, self).is_applicable_learnware(learnware):
            return False

        spec = learnware.get_specification()
        return is_hetero(stat_specs=spec.get_stat_spec(), semantic_spec=spec.get_semantic_spec(), verbose=False)

    def is_applicable_user(self, user_info: BaseUserInfo) -> bool:
        if not super(HeteroStatSearcher, self).is_applicable_user(user_info):
            return False

        stat_specs = user_info.stat_info
        semantic_spec = user_info.semantic_spec
        return is_hetero(stat_specs=stat_specs, semantic_spec=semantic_spec, verbose=False)

    def __call__(
        self,
        learnware_list: List[Learnware],
        user_info: BaseUserInfo,
        max_search_num: int = 5,
        search_method: str = "greedy",
    ) -> SearchResults:
        """Search learnwares based on user_info from learnwares with check_status.
           Employs heterogeneous learnware search if specific requirements are met, otherwise resorts to homogeneous search methods.

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        max_search_num : int
            The maximum number of the returned learnwares
        check_status : int, optional
            - None: search from all learnwares
            - Others: search from learnwares with check_status

        Returns
        -------
        Tuple[List[float], List[Learnware], float, List[Learnware]]
            the first is the sorted list of rkme dist
            the second is the sorted list of Learnware (single) by the rkme dist
            the third is the score of Learnware (mixture)
            the fourth is the list of Learnware (mixture), the size is search_num
        """
        user_hetero_spec = self.learnware_organizer.generate_hetero_map_spec(user_info)
        user_info.update_stat_info(user_hetero_spec.type, user_hetero_spec)

        return super().__call__(learnware_list, user_info, max_search_num, search_method)
