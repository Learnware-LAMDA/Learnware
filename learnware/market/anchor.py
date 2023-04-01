import os
from typing import Tuple, Any, List, Union, Dict

from .base import LearnwareMarket
from ..learnware import Learnware


class UserInfo:
    """Record the user info
    """

    def __init__(self, id: str, desc_info: dict, stat_info: dict = dict()):
        self.id = id
        self.desc_info = desc_info
        self.stat_info = stat_info
        self.anchor_learnware_list = {} # id: Learnware
    
    def get_desc_info(self):
        return self.desc_info
    
    def get_stat_info(self, name: str):
        return self.stat_info.get(name, None)
    
    def update_stat_info(self, name: str, item: Any):
        self.stat_info[name] = item
    


class AnchorMarket(BaseMarket):
    """Add the anchor design to the BaseMarket
    
    .. code-block:: python

        # Provide some python examples
        learnmarket = AnchorMarket()
    """
    
    def __init__(self):
        pass