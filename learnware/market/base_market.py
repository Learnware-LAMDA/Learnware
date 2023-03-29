import numpy as np
import pandas as pd
from typing import Tuple


class LearnwareMarket:
    def __init__(self):
        self.learnware_list = []
        self.count = 0

    def reload_market(self, market_path:str, property_list_path:str, load_mode:str = 'database')->bool:
        """
        Reload the market when server restared.

        Args:
            market_path (str): Directory for market data. '_IP_:_port_' for loading from database.
            property_list_path (str): Directory for available properties. Should be a json file.
            load_mode (str, optional): Type of reload source. Currently, only 'database' is available. Defaults to 'database'.

        Raises:
            NotImplemented: Reload method NOT implemented. Currently, only loading from database is supported.
            FileNotFoundError: Loading source/property_list NOT found. Check whether the source and property_list are available .

        Returns:
            bool: A flag indicating whether the market is reload successfully.
        """

        if load_mode=='database':
            pass
        else:
            # May Support other loading methods in the future
            raise NotImplemented("reload_market from {} is NOT implemented".format(load_mode))
        
        raise FileNotFoundError("Reload source NOT Found!")
        
        return True

    def add_learnware(self, learnware_name:str, model_path:str, )->Tuple[str, bool]:
        raise NotImplemented("add_learnware is not implemented")

    def search_learnware(self):
        raise NotImplemented("search_learnwars is not implemented")

    def get_learnware_by_ids(self):
        raise NotImplemented("get_learnware_by_ids is not implemented")

    def delete_learnware(self):
        raise NotImplemented("delete_learnware is not implemented")

    def get_sematic_list(self):
        raise NotImplemented("get_sematic_list is not implemented")
