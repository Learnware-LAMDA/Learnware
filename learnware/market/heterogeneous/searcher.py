from typing import Optional

from ..base import BaseUserInfo, SearchResults
from ..easy import EasyStatSearcher
from ...logger import get_module_logger

logger = get_module_logger("hetero_searcher")


class HeteroStatSearcher(EasyStatSearcher):
    SPEC_TYPES = ["HeteroMapTableSpecification"]

    def is_applicable_user(self, user_info: BaseUserInfo, verbose: bool = True) -> bool:
        stat_specs = user_info.stat_info
        semantic_spec = user_info.semantic_spec
        try:
            table_stat_spec = stat_specs["RKMETableSpecification"]
            table_input_shape = table_stat_spec.get_z().shape[1]

            semantic_data_type = semantic_spec["Data"]["Values"]
            if len(semantic_data_type) > 0 and semantic_data_type != ["Table"]:
                logger.warning("User doesn't provide correct data type, it must be Table.")
                return False

            semantic_task_type = semantic_spec["Task"]["Values"]
            if len(semantic_task_type) > 0 and semantic_task_type not in [["Classification"], ["Regression"]]:
                logger.warning(
                    "User doesn't provide correct task type, it must be either Classification or Regression."
                )
                return False

            semantic_input_description = semantic_spec["Input"]
            semantic_description_dim = int(semantic_input_description["Dimension"])
            semantic_decription_feature_num = len(semantic_input_description["Description"])

            if semantic_decription_feature_num <= 0:
                if verbose:
                    logger.warning("At least one of Input.Description in semantic spec should be provides.")
                return False

            if table_input_shape != semantic_description_dim:
                if verbose:
                    logger.warning("User data feature dimensions mismatch with semantic specification.")
                return False

            return True
        except Exception as err:
            if verbose:
                logger.warning("Invalid heterogeneous search information provided.")
            return False

    def __call__(
        self,
        user_info: BaseUserInfo,
        check_status: Optional[int] = None,
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

        return super().__call__(user_info, check_status, max_search_num, search_method)
