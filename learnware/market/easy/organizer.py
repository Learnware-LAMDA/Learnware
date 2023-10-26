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

from ..base import LearnwareMarket, BaseUserInfo
from ..database_ops import DatabaseOperations

from ... import utils
from ...config import C as conf
from ...logger import get_module_logger
from ...learnware import Learnware, get_learnware_from_dirpath
from ...specification import RKMEStatSpecification, Specification

from ..base import LearnwareOrganizer, LearnwareChecker
from ...logger import get_module_logger

logger = get_module_logger("easy_organizer")


class EasyOrganizer(LearnwareOrganizer):
    
    def reset(self, market_id, rebuild=False):
        self.market_id = market_id
        self.reload_market(rebuild=rebuild)
    
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
    
    
    def add_learnware(self, zip_path: str, semantic_spec: dict, learnware_id: str = None, check: bool = False) -> Tuple[str, bool]:
        """Add a learnware into the market.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.


        Parameters
        ----------
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.

        Returns
        -------
        Tuple[str, int]
            - str indicating model_id
            - int indicating what the flag of learnware is added.

        """
        semantic_spec = copy.deepcopy(semantic_spec)

        if not os.path.exists(zip_path):
            logger.warning("Zip Path NOT Found! Fail to add learnware.")
            return None, self.INVALID_LEARNWARE

        try:
            if len(semantic_spec["Data"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Data.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Task"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Task.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Library"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Device.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Name"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please provide Name.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Description"]["Values"]) == 0 and len(semantic_spec["Scenario"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please provide Scenario or Description.")
                return None, self.INVALID_LEARNWARE
            if (
                semantic_spec["Data"]["Type"] != "Class"
                or semantic_spec["Task"]["Type"] != "Class"
                or semantic_spec["Library"]["Type"] != "Class"
                or semantic_spec["Scenario"]["Type"] != "Tag"
                or semantic_spec["Name"]["Type"] != "String"
                or semantic_spec["Description"]["Type"] != "String"
            ):
                logger.warning("Illegal semantic specification, please provide the right type.")
                return None, self.INVALID_LEARNWARE
        except:
            print(semantic_spec)
            logger.warning("Illegal semantic specification, some keys are missing.")
            return None, self.INVALID_LEARNWARE

        logger.info("Get new learnware from %s" % (zip_path))
        if learnware_id is not None:
            id = learnware_id
        else:
            id = "%08d" % (self.count)
        target_zip_dir = os.path.join(self.learnware_zip_pool_path, "%s.zip" % (id))
        target_folder_dir = os.path.join(self.learnware_folder_pool_path, id)
        copyfile(zip_path, target_zip_dir)

        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        logger.info("Learnware move to %s, and unzip to %s" % (target_zip_dir, target_folder_dir))

        try:
            new_learnware = get_learnware_from_dirpath(
                id=id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
            )
        except:
            try:
                os.remove(target_zip_dir)
                rmtree(target_folder_dir)
            except:
                pass
            return None, self.INVALID_LEARNWARE

        if new_learnware is None:
            return None, self.INVALID_LEARNWARE

        if check and self.checker
        
        self.dbops.add_learnware(
            id=id,
            semantic_spec=semantic_spec,
            zip_path=target_zip_dir,
            folder_path=target_folder_dir,
            use_flag=LearnwareChecker.USABLE_LEARWARE,
        )

        self.learnware_list[id] = new_learnware
        self.learnware_zip_list[id] = target_zip_dir
        self.learnware_folder_list[id] = target_folder_dir
        self.count += 1
        return id, LearnwareChecker.USABLE_LEARWARE
    
    def add_learnware(self, zip_path: str, semantic_spec: dict) -> Tuple[str, bool]:
        """Add a learnware into the market.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.


        Parameters
        ----------
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.

        Returns
        -------
        Tuple[str, int]
            - str indicating model_id
            - int indicating what the flag of learnware is added.

        """
        semantic_spec = copy.deepcopy(semantic_spec)

        if not os.path.exists(zip_path):
            logger.warning("Zip Path NOT Found! Fail to add learnware.")
            return None, self.INVALID_LEARNWARE

        try:
            if len(semantic_spec["Data"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Data.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Task"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Task.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Library"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Device.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Name"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please provide Name.")
                return None, self.INVALID_LEARNWARE
            if len(semantic_spec["Description"]["Values"]) == 0 and len(semantic_spec["Scenario"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please provide Scenario or Description.")
                return None, self.INVALID_LEARNWARE
            if (
                semantic_spec["Data"]["Type"] != "Class"
                or semantic_spec["Task"]["Type"] != "Class"
                or semantic_spec["Library"]["Type"] != "Class"
                or semantic_spec["Scenario"]["Type"] != "Tag"
                or semantic_spec["Name"]["Type"] != "String"
                or semantic_spec["Description"]["Type"] != "String"
            ):
                logger.warning("Illegal semantic specification, please provide the right type.")
                return None, self.INVALID_LEARNWARE
        except:
            logger.info(f"Semantic specification: {semantic_spec}")
            logger.warning("Illegal semantic specification, some keys are missing.")
            return None, self.INVALID_LEARNWARE

        logger.info("Get new learnware from %s" % (zip_path))
        id = "%08d" % (self.count)
        target_zip_dir = os.path.join(self.learnware_zip_pool_path, "%s.zip" % (id))
        target_folder_dir = os.path.join(self.learnware_folder_pool_path, id)
        copyfile(zip_path, target_zip_dir)

        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        logger.info("Learnware move to %s, and unzip to %s" % (target_zip_dir, target_folder_dir))

        try:
            new_learnware = get_learnware_from_dirpath(
                id=id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
            )
        except:
            try:
                os.remove(target_zip_dir)
                rmtree(target_folder_dir)
            except:
                pass
            return None, self.INVALID_LEARNWARE

        if new_learnware is None:
            return None, self.INVALID_LEARNWARE

        check_flag = self.check_learnware(new_learnware)

        self.dbops.add_learnware(
            id=id,
            semantic_spec=semantic_spec,
            zip_path=target_zip_dir,
            folder_path=target_folder_dir,
            use_flag=check_flag,
        )

        self.learnware_list[id] = new_learnware
        self.learnware_zip_list[id] = target_zip_dir
        self.learnware_folder_list[id] = target_folder_dir
        self.count += 1
        return id, check_flag
    

    def get_learnware_ids(self, top:int = None):
        if top is None:
            return list(self.learnware_list.keys())
        else:
            return list(self.learnware_list.keys())[:top]
        
    
    def get_learnwares(self, top:int = None):
        if top is None:
            return list(self.learnware_list.values())
        else:
            return list(self.learnware_list.values())[:top]