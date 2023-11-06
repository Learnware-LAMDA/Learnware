from typing import List

from .organizer import HeteroMapTableOrganizer
from ...learnware import Learnware
from ..base import BaseSearcher, BaseUserInfo
from ...logger import get_module_logger

logger = get_module_logger("hetero_searcher")

class HeteroMapTableSearcher(BaseSearcher):
    def __init__(self, organizer: HeteroMapTableOrganizer = None):
        super(HeteroMapTableSearcher, self).__init__(organizer)

    def __call__(self, user_info: BaseUserInfo, check_status: int = None) -> Learnware:
        # todo: use specially assigned search_gamma for calculating mmd dist
        learnware_list = self.learnware_oganizer.get_learnwares()
        target_learnware, min_dist = None, None
        user_hetero_spec = self.learnware_oganizer.generate_hetero_map_spec(user_info)
        for learnware in learnware_list.values():
            learnware_hetero_spec = learnware.specification.get_stat_spec_by_name("HeteroSpecification")
            mmd_dist = learnware_hetero_spec.dist(user_hetero_spec)
            if target_learnware is None or mmd_dist < min_dist:
                min_dist = mmd_dist
                target_learnware = learnware
        return target_learnware
    
    def reset(self, organizer):
        self.learnware_oganizer = organizer