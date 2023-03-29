import numpy as np
import pandas as pd
from typing import Tuple, Any, List, Union, Dict
import os
from ..learnware import BaseLearnware


class LearnwareMarket:
    def __init__(self):
        """
        Initializing an empty market
        """
        self.learnware_list = {}  # id:BaseLearnware
        self.count = 0
        self.property_list = None

    def reload_market(self, market_path: str, property_list_path: str, load_mode: str = "database") -> bool:
        """
            Reload the market when server restared.

        Parameters
        ----------
        market_path : str
            Directory for market data. '_IP_:_port_' for loading from database.
        property_list_path : str
            Directory for available properties. Should be a json file.
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

    def add_learnware(
        self, learnware_name: str, model_path: str, stat_spec_path: str, properties: dict, desc: str
    ) -> Tuple[str, bool]:
        """
            Add a learnware into the market.
            Market will pack contents into a BaseLearnware object and assign an id automatically.

        Parameters
        ----------
        learnware_name : str
            Name of new learnware.
        model_path : str
            Filepath for learnware model, a zipped file.
        stat_spec_path : str
            Filepath for statistical specification, a '.npy' file.
            How to pass parameters requires further discussion.
        properties : dict
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

    def search_learnware(
        self, target_properties: dict = None, target_stat_specification: str = None
    ) -> Tuple[Any, Dict[str : List[Any]]]:
        """
            Search Learnware based on properties and statistical specification.
            Return random learnwares when both target_properties and target_stat_specification is None,
            Search only based on properties when target_stat_specification is None,
            Filter through properties and rank according to statistical specification otherwise.

        Parameters
        ----------
        target_properties : dict, optional
            Properties selected by user, by default None
        target_stat_specification : str, optional
            statistical specification uploaded by user, by default None

        Returns
        -------
        Tuple[Any, Dict[str:List[Any]]]
            return two items:
            first is recommended combination, None when no recommended combination is calculated or statistical specification is not provided.
            second is a list of matched learnwares

        Raises
        ------
        FileNotFoundError
            Give file path is empty.
        """

        if not os.path.exists(target_stat_specification):
            raise FileNotFoundError(
                "Statistical Specification File NOT Found. Please check param 'target_stat_specification'."
            )
        return None, []

    def get_learnware_by_ids(self, id: Union[str, List[str]]) -> Union[BaseLearnware, List[BaseLearnware]]:
        """
            Get Learnware from market by id

        Parameters
        ----------
        id : Union[str, List[str]]
            Given one id or a list of ids as target.

        Returns
        -------
        Union[BaseLearnware, List[BaseLearnware]]
            Return a BaseLearnware object or a list of BaseLearnware objects based on the type of input param.
            The returned items are search results.
            'None' indicating the target id not found.
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
