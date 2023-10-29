from typing import List

from ..evolve import EvolvedOrganizer
from ..anchor import AnchoredOrganizer, AnchoredUserInfo
from ...logger import get_module_logger

logger = get_module_logger("evolve_anchor_organizer")


class EvolvedAnchoredOrganizer(AnchoredOrganizer, EvolvedOrganizer):
    """Organize learnwares and enable them to continuously evolve"""

    def __init__(self, *args, **kwargs):
        AnchoredOrganizer.__init__(self, *args, **kwargs)

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
