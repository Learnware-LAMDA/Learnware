from typing import Tuple, Any, List, Union, Dict

from ..learnware import Learnware
from ..specification import BaseStatSpecification
from .anchor import AnchoredUserInfo, AnchoredMarket


class EvolvedMarket(AnchoredMarket):
    """Organize learnwares and enable them to continuously evolve

    Parameters
    ----------
    AnchoredMarket : _type_
        Market version with anchors
    """

    def __init__(self):
        super(EvolvedMarket, self).__init__()

    def generate_stat_specification(self, learnware: Learnware) -> BaseStatSpecification:
        """Generate new statistical specification for learnwares

        Parameters
        ----------
        learnware : Learnware

        Returns
        -------
        BaseStatSpecification
            New statistical specification
        """
        pass

    def evolve_anchor_learnware_list(self, anchor_id_list: List[str]):
        """Enable anchor learnwares to evolve, e.g., new stat_spec

        Parameters
        ----------
        anchor_id_list : List[str]
            Id list for Anchor learnwares
        """
        pass

    def evolve_learnware_list(self, id_list: List[str]):
        """Enable learnwares to evolve, e.g., new stat_spec

        Parameters
        ----------
        id_list : List[str]
            Id list for learnwares
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
