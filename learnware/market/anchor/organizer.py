from typing import List, Dict, Tuple, Any

from ..easy.organizer import EasyOrganizer
from ...logger import get_module_logger
from ...learnware import Learnware
from ...specification import BaseStatSpecification

logger = get_module_logger("anchor_organizer")


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
