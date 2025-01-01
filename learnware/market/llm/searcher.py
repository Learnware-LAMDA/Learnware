from typing import List, Optional, Tuple, Union

import numpy as np

from learnware.learnware.base import Learnware
from learnware.specification.base import Specification

from ..base import BaseUserInfo, SearchResults, AtomicSearcher
from ...logger import get_module_logger

logger = get_module_logger("llm_searcher")


class LLMStatSearcher(AtomicSearcher):
    SPEC_TYPES = ["GenerativeModelSpecification"]

    def is_applicable_user(self, user_info: BaseUserInfo, verbose: bool = True) -> bool:
        stat_specs = user_info.stat_info
        semantic_spec = user_info.semantic_spec
        try:
            if "GenerativeModelSpecification" not in stat_specs:
                if verbose:
                    logger.warning("GenerativeModelSpecification is not provided in stat_info.")
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

    def _search_by_taskvector_spec_single(
        self,
        learnware_list: List[Learnware],
        user_spec: Union[Specification],
        stat_spec_type: str = "GenerativeModelSpecification"
    ) -> Tuple[List[float], List[Learnware]]:
        """Calculate the distances between learnwares in the given learnware_list and user_spec

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : Union[RKMETableSpecification, RKMEImageSpecification, RKMETextSpecification]
            user Task Vector statistical specification
        stat_spec_type : str
            GenerativeModelSpecification by default.

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            the first is the list of cosine similarity
            the second is the list of Learnware
            both lists are sorted by cosine similarity
        """
        spec_list = [learnware.specification.get_stat_spec_by_name(stat_spec_type) for learnware in learnware_list]
        filtered_idx_list, similarity_list = [], []
        for idx, s in enumerate(spec_list):
            similarity = float(s.similarity(user_spec))
            if np.isfinite(similarity):
                similarity_list.append(similarity)
                filtered_idx_list.append(idx)
            else:
                logger.warning(
                    f"The distance between user_spec and learnware_spec (id: {learnware_list[idx].id}) is not finite, where distance is {mmd_dist}"
                )

        sorted_idx_list = reversed(sorted(range(len(similarity_list)), key=lambda k: similarity_list[k]))
        sorted_dist_list = [similarity_list[idx] for idx in sorted_idx_list]
        sorted_learnware_list = [learnware_list[filtered_idx_list[idx]] for idx in sorted_idx_list]

        return sorted_dist_list, sorted_learnware_list