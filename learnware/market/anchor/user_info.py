from typing import List, Any, Union
from ..base import BaseUserInfo


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
        """Update stat_info by market or user with anchor learnwares

        Parameters
        ----------
        name : str
            Name of stat_info
        item : Any
            Statistical information calculated by market or user with anchor learnwares
        """
        self.stat_info[name] = item
