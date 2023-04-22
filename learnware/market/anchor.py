import os
from typing import Tuple, Any, List, Union, Dict

from ..learnware import Learnware
from .base import BaseMarket, BaseUserInfo


class AnchoredUserInfo(BaseUserInfo):
    """
    User Information for searching learnware (add the anchor design)

    - UserInfo contains the anchor list acquired from the market
    - UserInfo can update stat_info based on anchors
    """

    def __init__(self, id: str, semantic_spec: dict = dict(), stat_info: dict = dict()):
        super(AnchoredUserInfo, self).__init__(id, semantic_spec, stat_info)
        self.anchor_learnware_list = {}  # id: Learnware

    def add_anchor_learnware(self, learnware_id: str, learnware: Learnware):
        """Add the anchor learnware acquired from the market

        Parameters
        ----------
        learnware_id : str
            Id of anchor learnware
        learnware : Learnware
            Anchor learnware for capturing user requirements
        """
        self.anchor_learnware_list[learnware_id] = learnware

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


class AnchoredMarket(BaseMarket):
    """Add the anchor design to the BaseMarket

    Parameters
    ----------
    BaseMarket : _type_
        Basic market version
    """

    def __init__(self, *args, **kwargs):
        super(AnchoredMarket, self).__init__(*args, **kwargs)
        self.anchor_learnware_list = {}  # anchor_id: anchor learnware

    def _update_anchor_learnware(self, anchor_id: str, anchor_learnware: Learnware):
        """Update anchor_learnware_list

        Parameters
        ----------
        anchor_id : str
            Id of anchor learnware
        anchor_learnware : Learnware
            Anchor learnware
        """
        self.anchor_learnware_list[anchor_id] = anchor_learnware

    def _delete_anchor_learnware(self, anchor_id: str) -> bool:
        """Delete anchor learnware in anchor_learnware_list

        Parameters
        ----------
        anchor_id : str
            Id of anchor learnware

        Returns
        -------
        bool
            True if the target anchor learnware is deleted successfully.

        Raises
        ------
        Exception
            Raise an excpetion when given anchor_id is NOT found in anchor_learnware_list
        """
        if not anchor_id in self.anchor_learnware_list:
            raise Exception("Anchor learnware id:{} NOT Found!".format(anchor_id))

        self.anchor_learnware_list.pop(anchor_id)
        return True

    def update_anchor_learnware_list(self, learnware_list: Dict[str, Learnware]):
        """Update anchor_learnware_list

        Parameters
        ----------
        learnware_list : Dict[str, Learnware]
            Learnwares for updating anchor_learnware_list
        """
        pass

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
