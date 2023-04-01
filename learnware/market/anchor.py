import os
from typing import Tuple, Any, List, Union, Dict

from .base import BaseUserInfo, BaseMarket
from ..learnware import Learnware


class AnchoredUserInfo(BaseUserInfo):
    """
        User Information for searching learnware (add the anchor design)
        
        - UserInfo contains the anchor list acquired from the market
        - UserInfo can update stat_info based on anchors
    """

    def __init__(self, id: str, property: dict = dict(), stat_info: dict = dict()):
        super(AnchoredUserInfo, self).__init__(id, property, stat_info)
        self.anchor_learnware_list = {} # id: Learnware
    
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
    
    .. code-block:: python

        # Provide some python examples
        learnmarket = AnchoredMarket()
    """
    
    def __init__(self):
        super(AnchoredMarket, self).__init__()
    
    def get_anchor_learnware(self, user_info: AnchoredUserInfo) -> Tuple[Any, Dict[str, List[Any]]]:
        """Get anchor Learnware based on user_info

        Parameters
        ----------
        user_info : AnchoredUserInfo
            - user_info with properties and statistical information
            - some statistical information calculated on anchor learnwares

        Returns
        -------
        Tuple[Any, Dict[str, List[Any]]]
            return two items:

            - first is recommended combination, None when no recommended combination is calculated or statistical specification is not provided.
            - second is a list of matched learnwares
        """
        pass
    
    def search_learnware(self, user_info: AnchoredUserInfo) -> Tuple[Any, Dict[str, List[Any]]]:
        """Search learnware based on user_info

        Parameters
        ----------
        user_info : AnchoredUserInfo
            - user_info with properties and statistical information
            - some statistical information calculated on anchor learnwares

        Returns
        -------
        Tuple[Any, Dict[str, List[Any]]]
            return two items:

            - first is recommended combination, None when no recommended combination is calculated or statistical specification is not provided.
            - second is a list of matched learnwares
        """
        pass