from typing import List

from ..easy.organizer import EasyOrganizer
from ...learnware import Learnware
from ...specification import BaseStatSpecification
from ...logger import get_module_logger

logger = get_module_logger("evolve_organizer")


class EvolvedOrganizer(EasyOrganizer):
    """Organize learnwares and enable them to continuously evolve"""

    def __init__(self, *args, **kwargs):
        super(EvolvedOrganizer, self).__init__(*args, **kwargs)

    def generate_new_stat_specification(self, learnware: Learnware) -> BaseStatSpecification:
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

    def evolve_learnware_list(self, id_list: List[str]):
        """Enable learnwares to evolve, e.g., new stat_spec

        Parameters
        ----------
        id_list : List[str]
            Id list for learnwares
        """
        pass
