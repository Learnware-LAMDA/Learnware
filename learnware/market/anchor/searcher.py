from typing import List, Dict, Tuple, Any, Union

from .user_info import AnchoredUserInfo
from ..base import BaseUserInfo
from ..easy.searcher import EasySearcher
from ...logger import get_module_logger
from ...learnware import Learnware

logger = get_module_logger("anchor_searcher")


class AnchoredSearcher(EasySearcher):
    def search_anchor_learnware(self, user_info: AnchoredUserInfo) -> Tuple[Any, List[Learnware]]:
        """Search anchor Learnwares from anchor_learnware_list based on user_info

        Parameters
        ----------
        user_info : AnchoredUserInfo
            - user_info with semantic specifications and statistical information
            - some statistical information calculated on previous anchor learnwares

        Returns
        -------
        Tuple[Any, List[Learnware]]:
            return two items:

            - first is the usage of anchor learnwares, e.g., how to use anchors to calculate some statistical information
            - second is a list of anchor learnwares
        """
        pass

    def search_learnware(self, user_info: AnchoredUserInfo) -> Tuple[Any, List[Learnware]]:
        """Find helpful learnwares from learnware_list based on user_info

        Parameters
        ----------
        user_info : AnchoredUserInfo
            - user_info with semantic specifications and statistical information
            - some statistical information calculated on anchor learnwares

        Returns
        -------
        Tuple[Any, List[Any]]
            return two items:

            - first is recommended combination, None when no recommended combination is calculated or statistical specification is not provided.
            - second is a list of matched learnwares
        """
        pass

    def __call__(self, user_info: AnchoredUserInfo, anchor_flag: bool = False) -> Tuple[Any, List[Learnware]]:
        """Search learnwares with anchor marget
        - if 'anchor_flag' == True, search anchor Learnwares from anchor_learnware_list based on user_info
        - if 'anchor_flag' == False, find helpful learnwares from learnware_list based on user_info

        Parameters
        ----------
        user_info : AnchoredUserInfo
            - user_info with semantic specifications and statistical information
            - some statistical information calculated on anchor learnwares

        Returns
        -------
        Tuple[Any, List[Any]]
            return two items:

            - first is recommended combination, None when no recommended combination is calculated or statistical specification is not provided.
            - second is a list of matched learnwares
        """
        if anchor_flag:
            return self.search_anchor_learnware(user_info)
        else:
            return self.search_learnware(user_info)
