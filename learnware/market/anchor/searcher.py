from typing import List, Dict, Tuple, Any, Union

from ..base import BaseUserInfo
from ..easy2.searcher import EasySearcher
from ...logger import get_module_logger
from ...learnware import Learnware

logger = get_module_logger("anchor_searcher")


class AnchoredUserInfo(BaseUserInfo):
    """
    User Information for searching learnware (add the anchor design)

    - UserInfo contains the anchor id list acquired from the market
    - UserInfo can update stat_info based on anchors
    """

    def __init__(
        self, id: str, semantic_spec: dict = None, stat_info: dict = None, anchor_learnware_ids: List[str] = None
    ):
        super(AnchoredUserInfo, self).__init__(id, semantic_spec, stat_info)
        self.anchor_learnware_ids = [] if anchor_learnware_ids is None else anchor_learnware_ids

    def add_anchor_learnware_ids(self, learnware_ids: Union[str, List[str]]):
        """Add the anchor learnware ids acquired from the market

        Parameters
        ----------
        learnware_ids : Union[str, List[str]]
            Anchor learnware ids
        """
        if isinstance(learnware_ids, str):
            learnware_ids = [learnware_ids]
        self.anchor_learnware_ids += learnware_ids

    def update_stat_info(self, name: str, item: Any):
        """Update stat_info based on anchor learnwares

        Parameters
        ----------
        name : str
            Name of stat_info
        item : Any
            Statistical information calculated on anchor learnwares
        """
        self.stat_info[name] = item


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
