import os
import json
import copy
import torch
import zipfile
import traceback
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from cvxopt import solvers, matrix
from shutil import copyfile, rmtree
from typing import Tuple, Any, List, Union, Dict

from ..base import BaseMarket, BaseUserInfo
from ..database_ops import DatabaseOperations

from ... import utils
from ...config import C as conf
from ...logger import get_module_logger
from ...learnware import Learnware, get_learnware_from_dirpath
from ...specification import RKMEStatSpecification, Specification

from ..base import LearnwareOrganizer
from ...logger import get_module_logger

logger = get_module_logger("easy_organizer")


class EasyOrganizer(LearnwareOrganizer):
    
    def reset(self, market_id):
        self.market_id = market_id
        self.reload_market()
    
    def reload_market(self, rebuild=False) -> bool:
        """Reload the learnware organizer when server restared.
        
        Returns
        -------
        bool
            A flag indicating whether the market is reload successfully.
        """
 
        self.market_store_path = os.path.join(conf.market_root_path, self.market_id)
        self.learnware_pool_path = os.path.join(self.market_store_path, "learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.learnware_pool_path, "zips")
        self.learnware_folder_pool_path = os.path.join(self.learnware_pool_path, "unzipped_learnwares")
        self.learnware_list = {}  # id: Learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.count = 0
        self.semantic_spec_list = conf.semantic_specs
        self.dbops = DatabaseOperations(conf.database_url, "market_" + self.market_id)
        
        if rebuild:
            logger.warning("Warning! You are trying to clear current database!")
            try:
                self.dbops.clear_learnware_table()
                rmtree(self.learnware_pool_path)
            except:
                pass

        os.makedirs(self.learnware_pool_path, exist_ok=True)
        os.makedirs(self.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(self.learnware_folder_pool_path, exist_ok=True)
        self.learnware_list, self.learnware_zip_list, self.learnware_folder_list, self.count = self.dbops.load_market()
        
        