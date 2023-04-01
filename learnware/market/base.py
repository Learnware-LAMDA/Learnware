import os
import numpy as np
import pandas as pd
from typing import Tuple, Any, List, Union, Dict

from ..learnware import Learnware


class BaseUserInfo:
    """
        User Information for searching learnware
        
        - Return random learnwares when both property and stat_info is empty
        - Search only based on property when stat_info is None
        - Filter through property and rank according to stat_info otherwise
    """

    def __init__(self, id: str, property: dict = dict(), stat_info: dict = dict()):
        """Initializing user information

        Parameters
        ----------
        id : str
            user id
        property : dict, optional
            property selected by user, by default dict()
        stat_info : dict, optional
            statistical information uploaded by user, by default dict()
        """
        self.id = id
        self.property = property
        self.stat_info = stat_info
    
    def get_property(self) -> dict:
        """Return user properties

        Returns
        -------
        dict
            user properties
        """
        return self.property
    
    def get_stat_info(self, name: str):
        return self.stat_info.get(name, None)


class BaseLearnwareMarket:
    """Market for Learnware

    .. code-block:: python

        # Provide some python examples
        learnmarket = LearnwareMarket()
    """

    def __init__(self):
        """Initializing an empty market"""
        self.learnware_list = {}  # id: Learnware
        self.count = 0
        self.property_list = None

    def reload_market(self, market_path: str, property_list_path: str, load_mode: str = "database") -> bool:
        """Reload the market when server restared.

        Parameters
        ----------
        market_path : str
            Directory for market data. '_IP_:_port_' for loading from database.
        property_list_path : str
            Directory for available property. Should be a json file.
        load_mode : str, optional
            Type of reload source. Currently, only 'database' is available. Defaults to 'database', by default "database"

        Returns
        -------
        bool
            A flag indicating whether the market is reload successfully.

        Raises
        ------
        NotImplementedError
            Reload method NOT implemented. Currently, only loading from database is supported.
        FileNotFoundError
            Loading source/property_list NOT found. Check whether the source and property_list are available.

        """

        if load_mode == "database":
            pass
        else:
            # May Support other loading methods in the future
            raise NotImplementedError("reload_market from {} is NOT implemented".format(load_mode))

        raise FileNotFoundError("Reload source NOT Found!")

        return True

    def check_learnware(self, learnware: Learnware) -> bool:
        """Check the utility of a learnware
        
        Parameters
        ----------
        learnware : Learnware

        Returns
        -------
        bool
            A flag indicating whether the learnware can be accepted.
        """
        return True
    
    def add_learnware(
        self, learnware_name: str, model_path: str, stat_spec_path: str, property: dict, desc: str
    ) -> Tuple[str, bool]:
        """Add a learnware into the market.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.


        Parameters
        ----------
        learnware_name : str
            Name of new learnware.
        model_path : str
            Filepath for learnware model, a zipped file.
        stat_spec_path : str
            Filepath for statistical specification, a '.npy' file.
            How to pass parameters requires further discussion.
        property : dict
            property for new learnware, in dictionary format.
        desc : str
            Brief desciption for new learnware.

        Returns
        -------
        Tuple[str, bool]
            str indicating model_id, bool indicating whether the learnware is added successfully.

        Raises
        ------
        FileNotFoundError
            file for model or statistical specification not found

        """
        if (not os.path.exists(model_path)) or (not os.path.exists(stat_spec_path)):
            raise FileNotFoundError("Model or Stat_spec NOT Found.")
        return str(self.count), True

    def search_learnware(self, user_info: BaseUserInfo) -> Tuple[Any, Dict[str, List[Any]]]:
        """Search Learnware based on user_info

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info with properties and statistical information

        Returns
        -------
        Tuple[Any, Dict[str, List[Any]]]
            return two items:

            - first is recommended combination, None when no recommended combination is calculated or statistical specification is not provided.
            - second is a list of matched learnwares
        """
        pass

    def get_learnware_by_ids(self, id: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """
            Get Learnware from market by id

        Parameters
        ----------
        id : Union[str, List[str]]
            Given one id or a list of ids as target.

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return a Learnware object or a list of Learnware objects based on the type of input param.

            - The returned items are search results.
            - 'None' indicating the target id not found.
        """
        return None

    def delete_learnware(self, id: str) -> bool:
        """
            deleted a learnware from market

        Parameters
        ----------
        id : str
            id of learnware to be deleted

        Returns
        -------
        bool
            True if the target learnware is deleted successfully.

        Raises
        ------
        Exception
            Raise an excpetion when give id is NOT found in learnware list
        """
        if not id in self.learnware_list:
            raise Exception("Learnware id:{} NOT Found!".format(id))
        return True

    def update_learnware(self, id: str) -> bool:
        """
            Update Learnware with id and content to be updated.
            Empty interface. TODO

        Parameters
        ----------
        id : str
            id of target learnware.
        """
        return True

    def get_property_list(self) -> dict:
        """Return all properties available

        Returns
        -------
        dict
            All properties in dictionary format

        """
        return self.property_list
