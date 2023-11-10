from typing import Tuple, List, Union

import numpy as np

from ...learnware import Learnware
from ...logger import get_module_logger
from ...specification import HeteroSpecification
from ..base import BaseSearcher, BaseUserInfo
from ..easy import EasySearcher
from ..utils import parse_specification_type
from .organizer import HeteroMapTableOrganizer

logger = get_module_logger("hetero_searcher")


class HeteroMapTableSearcher(EasySearcher):
    def _convert_dist_to_score(
        self, dist_list: List[float], dist_epsilon: float = 0.01, min_score: float = 0.92
    ) -> List[float]:
        if len(dist_list) == 0:
            return []

        min_dist, max_dist = min(dist_list), max(dist_list)
        if min_dist == max_dist:
            return [1 for dist in dist_list]
        else:
            max_score = (max_dist - min_dist) / (max_dist - dist_epsilon)

            if min_dist < dist_epsilon:
                dist_epsilon = min_dist
            elif max_score < min_score:
                dist_epsilon = max_dist - (max_dist - min_dist) / min_score

            return [(max_dist - dist) / (max_dist - dist_epsilon) for dist in dist_list]

    def _search_by_hetero_spec_single(
        self, 
        learnware_list: List[Learnware], 
        user_hetero_spec: HeteroSpecification
    ) -> Tuple[List[float], List[Learnware]]:
        hetero_spec_list = [learnware.specification.get_stat_spec_by_name("HeteroSpecification") for learnware in learnware_list]
        mmd_dist_list = []
        for idx, hetero_spec in enumerate(hetero_spec_list):
            mmd_dist = hetero_spec.dist(user_hetero_spec)
            mmd_dist_list.append(mmd_dist)
        
        sorted_idx_list = sorted(range(len(learnware_list)), key=lambda k: mmd_dist_list[k])
        sorted_dist_list = [mmd_dist_list[idx] for idx in sorted_idx_list]
        sorted_learnware_list = [learnware_list[idx] for idx in sorted_idx_list]

        return sorted_dist_list, sorted_learnware_list
    
    def _filter_by_hetero_spec_single(
        self,
        sorted_score_list: List[float],
        learnware_list: List[Learnware],
        filter_score: float = 0.5,
        min_num: int = 5
    ) -> Tuple[List[float], List[Learnware]]:
        idx = min(min_num, len(learnware_list))
        while idx < len(learnware_list):
            if sorted_score_list[idx] < filter_score:
                break
            idx += 1
        return sorted_score_list[:idx], learnware_list[:idx]


    def __call__(
        self, 
        learnware_list: List[Learnware], 
        user_info: BaseUserInfo, 
    ) -> Tuple[List[float], List[Learnware], float, List[Learnware]]:
        # todo: use specially assigned search_gamma for calculating mmd dist
        user_hetero_spec = self.learnware_oganizer.generate_hetero_map_spec(user_info)
        logger.info(f"After semantic search, learnware_list length is {len(learnware_list)}")

        sorted_dist_list, single_learnware_list = self._search_by_hetero_spec_single(learnware_list, user_hetero_spec)
        sorted_score_list = self._convert_dist_to_score(sorted_dist_list)

        logger.info(f"After search by hetero spec, learnware_list length is {len(single_learnware_list)}")
        sorted_score_list, single_learnware_list = self._filter_by_hetero_spec_single(
            sorted_score_list, single_learnware_list
        )

        logger.info(f"After filter by hetero spec, learnware_list length is {len(single_learnware_list)}")
        return sorted_score_list, single_learnware_list, None, None

    def reset(self, organizer):
        self.learnware_oganizer = organizer

class HeteroSearcher(EasySearcher):
    def __init__(self, organizer: HeteroMapTableOrganizer = None):
        super(HeteroSearcher, self).__init__(organizer)
        self.hetero_stat_searcher = HeteroMapTableSearcher(organizer)

    def reset(self, organizer):
        super().reset(organizer)
        self.hetero_stat_searcher.reset(organizer)
    
    @staticmethod
    def check_user_info(user_info: BaseUserInfo):
        try:
            user_stat_spec = user_info.get_stat_info("RKMETableSpecification")
            user_input_shape = user_stat_spec.get_z().shape[1]

            user_task_type = user_info.get_semantic_spec().get("Task", {}).get("Values")
            if user_task_type not in [["Classification"], ["Regression"]]:
                logger.warning("User doesn't provide correct task type, it must be either Classification or Regression.")
                return False

            user_input_description = user_info.get_semantic_spec().get("Input", {})
            user_description_dim = int(user_input_description.get("Dimension", 0))
            user_description_feature_num = len(user_input_description.get("Description", []))

            if user_input_shape != user_description_dim or user_input_shape != user_description_feature_num:
                logger.warning("User data feature dimensions mismatch with semantic specification.")
                return False
            
            return True
        except Exception as e:
            logger.info(f"Invalid heterogeneous search information provided. Use homogeneous search instead. Error: {e}")
            return False

    def __call__(
        self, user_info: BaseUserInfo, check_status: int = None, max_search_num: int = 5, search_method: str = "greedy"
    ) -> Tuple[List[float], List[Learnware], float, List[Learnware]]:
        learnware_list = self.learnware_organizer.get_learnwares(check_status=check_status)
        learnware_list = self.semantic_searcher(learnware_list, user_info)

        if len(learnware_list) == 0:
            return [], [], 0.0, []

        if parse_specification_type(stat_specs=user_info.stat_info) is not None:
            if self.check_user_info(user_info):
                return self.hetero_stat_searcher(learnware_list, user_info)
            else:
                return self.stat_searcher(learnware_list, user_info, max_search_num, search_method)
        else:
            return None, learnware_list, 0.0, None