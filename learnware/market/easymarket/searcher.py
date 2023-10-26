from typing import Tuple, List

from ..base import LearnwareSearcher
from ...logger import get_module_logger
from ...learnware import Learnware
from ...market import BaseUserInfo

logger = get_module_logger('easy_seacher')

class EasySearcher(LearnwareSearcher):
    
    def __call__(self, user_info: BaseUserInfo, max_search_num: int = 5, search_method: str = "greedy") -> Tuple[List[float], List[Learnware], float, List[Learnware]]:
        """Search learnwares based on user_info

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        max_search_num : int
            The maximum number of the returned learnwares

        Returns
        -------
        Tuple[List[float], List[Learnware], float, List[Learnware]]
            the first is the sorted list of rkme dist
            the second is the sorted list of Learnware (single) by the rkme dist
            the third is the score of Learnware (mixture)
            the fourth is the list of Learnware (mixture), the size is search_num
        """
        learnware_list = [self.learnware_list[key] for key in self.learnware_list]
        # learnware_list = self._search_by_semantic_spec_exact(learnware_list, user_info)
        # if len(learnware_list) == 0:
        learnware_list = self._search_by_semantic_spec_fuzz(learnware_list, user_info)

        if "RKMEStatSpecification" not in user_info.stat_info:
            return None, learnware_list, 0.0, None
        elif len(learnware_list) == 0:
            return [], [], 0.0, []
        else:
            user_rkme = user_info.stat_info["RKMEStatSpecification"]
            learnware_list = self._filter_by_rkme_spec_dimension(learnware_list, user_rkme)
            logger.info(f"After filter by rkme dimension, learnware_list length is {len(learnware_list)}")

            sorted_dist_list, single_learnware_list = self._search_by_rkme_spec_single(learnware_list, user_rkme)
            if search_method == "auto":
                mixture_dist, weight_list, mixture_learnware_list = self._search_by_rkme_spec_mixture_auto(
                    learnware_list, user_rkme, max_search_num
                )
            elif search_method == "greedy":
                mixture_dist, weight_list, mixture_learnware_list = self._search_by_rkme_spec_mixture_greedy(
                    learnware_list, user_rkme, max_search_num
                )
            else:
                logger.warning("f{search_method} not supported!")
                mixture_dist = None
                weight_list = []
                mixture_learnware_list = []

            if mixture_dist is None:
                sorted_score_list = self._convert_dist_to_score(sorted_dist_list)
                mixture_score = None
            else:
                merge_score_list = self._convert_dist_to_score(sorted_dist_list + [mixture_dist])
                sorted_score_list = merge_score_list[:-1]
                mixture_score = merge_score_list[-1]

            logger.info(f"After search by rkme spec, learnware_list length is {len(learnware_list)}")
            # filter learnware with low score
            sorted_score_list, single_learnware_list = self._filter_by_rkme_spec_single(
                sorted_score_list, single_learnware_list
            )

            logger.info(f"After filter by rkme spec, learnware_list length is {len(learnware_list)}")
            return sorted_score_list, single_learnware_list, mixture_score, mixture_learnware_list
    