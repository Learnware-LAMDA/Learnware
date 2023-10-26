from typing import List, Dict, Tuple, Any

from ..base import BaseUserInfo
from ..easy2.organizer import EasyOrganizer
from ...logger import get_module_logger
from ...learnware import Learnware
from ...specification import BaseStatSpecification

logger = get_module_logger("evolve_organizer")


class AnchoredUserInfo(BaseUserInfo):
    """
    User Information for searching learnware (add the anchor design)

    - UserInfo contains the anchor list acquired from the market
    - UserInfo can update stat_info based on anchors
    """

    def __init__(self, id: str, semantic_spec: dict = None, stat_info: dict = None, anchor_scores: dict = None):
        super(AnchoredUserInfo, self).__init__(id, semantic_spec, stat_info)
        self.anchor_scores = {} if anchor_scores is None else anchor_scores

    def update_anchor_score(self, id: str, score):
        """Update score of anchor learnwares

        Parameters
        ----------
        id : str
            id of anchor learnwares
        score : Any
            score of anchor learnwares
        """
        self.anchor_scores[id] = score


class AnchoredOrganizer(EasyOrganizer):
    """Organize learnwares and enable them to continuously evolve"""

    def __init__(self, *args, **kwargs):
        super(AnchoredOrganizer, self).__init__(*args, **kwargs)
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

    def search_learnware(self, user_info: AnchoredUserInfo, anchored: bool = False) -> Tuple[Any, List[Learnware]]:
        """Search learnwares with anchor marget
        - if 'anchor' == True, search anchor Learnwares from anchor_learnware_list based on user_info
        - if 'anchor' == False, find helpful learnwares from learnware_list based on user_info

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
