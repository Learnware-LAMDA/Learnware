from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from learnware.learnware.base import Learnware
from learnware.specification.base import Specification
from ..utils import parse_specification_type
from ..base import BaseUserInfo, MultipleSearchItem, SearchResults, AtomicSearcher, SingleSearchItem
from ..easy import EasyStatSearcher
from ...logger import get_module_logger

from torch.nn.functional import softmax

logger = get_module_logger("llm_searcher")


class LLMStatSearcher(EasyStatSearcher):
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
        learnware_list: List[Learnware],
        user_info: BaseUserInfo,
        max_search_num: int = 5,
        search_method: str = "greedy",
    ) -> SearchResults:
        self.stat_spec_type = parse_specification_type(stat_specs=user_info.stat_info, spec_list=self.SPEC_TYPES)

        user_spec = user_info.stat_info[self.stat_spec_type]

        sorted_metric_list, single_learnware_list = self._search_by_taskvector_spec_single(learnware_list, user_spec)
        if len(single_learnware_list) == 0:
            return SearchResults()

        if self.stat_spec_type == "GenerativeModelSpecification":
            sorted_score_list = self._convert_similarity_to_score(sorted_metric_list)
        else:
            sorted_score_list = self._convert_dist_to_score(sorted_metric_list)
        
        logger.info(
            f"After search by user spec, learnware_list length is {len(learnware_list)}"
        )

        if len(single_learnware_list) == 1 and sorted_score_list[0] < 0.6:
            sorted_score_list[0] = 0.6

        search_results = SearchResults()
        search_results.update_single_results(
            [
                SingleSearchItem(learnware=_learnware, score=_score)
                for _score, _learnware in zip(sorted_score_list, single_learnware_list)
            ]
        )

        return search_results

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
        user_spec : GenerativeModelSpecification
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
            user_spec.task_vector = user_spec.task_vector.to(s.task_vector.device)
            similarity = float(s.similarity(user_spec))
            if np.isfinite(similarity):
                similarity_list.append(similarity)
                filtered_idx_list.append(idx)
            else:
                logger.warning(
                    f"The distance between user_spec and learnware_spec (id: {learnware_list[idx].id}) is not finite, where similarity is {similarity}"
                )

        sorted_idx_list = list(reversed(sorted(range(len(similarity_list)), key=lambda k: similarity_list[k])))
        sorted_dist_list = [similarity_list[idx] for idx in sorted_idx_list]
        sorted_learnware_list = [learnware_list[filtered_idx_list[idx]] for idx in sorted_idx_list]

        return sorted_dist_list, sorted_learnware_list
    
    def _convert_similarity_to_score(self, sorted_similarity_list, temperature=0.1):
        sorted_similarity = torch.asarray(sorted_similarity_list)
        sorted_similarity = torch.stack([
            sorted_similarity, torch.zeros_like(sorted_similarity)
        ])
        
        scores = softmax(sorted_similarity / temperature, dim=0)[0].tolist()
        return scores * 100