from typing import Tuple, Any, List, Union, Dict

from .base import BaseMarket
from ..learnware import Learnware
from ..specification import BaseStatSpecification


class EvolvedMarket(BaseMarket):
    """Organize learnwares and enable them to continuously evolve

    Parameters
    ----------
    BaseMarket : _type_
        Basic market version
    """

    def __init__(self, *args, **kwargs):
        super(EvolvedMarket, self).__init__(*args, **kwargs)

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
