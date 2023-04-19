from typing import Tuple, Any, List, Union, Dict

from .anchor import AnchoredUserInfo, AnchoredMarket
from .evolve import EvolvedMarket


class EvolvedAnchoredMarket(AnchoredMarket, EvolvedMarket):
    """Organize learnwares with anchors and enable them to continuously evolve

    Parameters
    ----------
    AnchoredMarket : _type_
        Market version with anchors
    EvolvedMarket : _type_
        Market version with evolved learnwares
    """

    def __init__(self):
        super(EvolvedAnchoredMarket, self).__init__()

    def evolve_anchor_learnware_list(self, anchor_id_list: List[str]):
        """Enable anchor learnwares to evolve, e.g., new stat_spec

        Parameters
        ----------
        anchor_id_list : List[str]
            Id list for Anchor learnwares
        """
        pass

    def evolve_anchor_learnware_by_user(self, user_info: AnchoredUserInfo):
        """Enable anchor leanrwares to evolve based on user statistical information

        Parameters
        ----------
        user_info : AnchoredUserInfo
            User information with statistics calculated on anchors
        """
        pass

    def evolve_learnware_by_user(self, learnware_id: str, user_info: AnchoredUserInfo):
        """
            Enable leanrwares to evolve based on user info
            - e.g., When we estimate the performance of a specific learnware on user tasks, we can further refine the update of the learnware specification

        Parameters
        ----------
        learnware_id : str
            Leanrware id
        user_info : AnchoredUserInfo
            User information with statistics calculated on anchors
        """
        pass
