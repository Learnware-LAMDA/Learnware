from typing import Tuple, List
import traceback
from ...learnware import Learnware
from ...logger import get_module_logger
from ..base import BaseUserInfo
from ..easy import EasySearcher
from ..utils import parse_specification_type

logger = get_module_logger("hetero_searcher")


class HeteroSearcher(EasySearcher):
    @staticmethod
    def check_user_info(user_info: BaseUserInfo):
        try:
            user_stat_spec = user_info.get_stat_info("RKMETableSpecification")
            user_input_shape = user_stat_spec.get_z().shape[1]

            user_task_type = user_info.get_semantic_spec()["Task"]["Values"]
            if user_task_type not in [["Classification"], ["Regression"]]:
                logger.warning(
                    "User doesn't provide correct task type, it must be either Classification or Regression."
                )
                return False

            user_input_description = user_info.get_semantic_spec()["Input"]
            user_description_dim = int(user_input_description["Dimension"])
            user_description_feature_num = len(user_input_description["Description"])

            if user_input_shape != user_description_dim or user_input_shape != user_description_feature_num:
                logger.warning("User data feature dimensions mismatch with semantic specification.")
                return False

            return True

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Invalid heterogeneous search information provided. Use homogeneous search instead.")
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
                user_hetero_spec = self.learnware_organizer.generate_hetero_map_spec(user_info)
                user_info.update_stat_info(user_hetero_spec.type, user_hetero_spec)
            return self.stat_searcher(learnware_list, user_info, max_search_num, search_method)
        else:
            return None, learnware_list, 0.0, None
