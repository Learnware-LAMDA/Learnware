import os
import json
import copy
import torch
import zipfile
import traceback
import tempfile
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from cvxopt import solvers, matrix
from shutil import copyfile, rmtree
from typing import Tuple, Any, List, Union, Dict

from .database_ops import DatabaseOperations
from ..base import LearnwareMarket, BaseUserInfo


from ... import utils
from ...config import C as conf
from ...logger import get_module_logger
from ...learnware import Learnware, get_learnware_from_dirpath
from ...specification import RKMEStatSpecification, Specification

from ..base import BaseOrganizer, BaseChecker
from ...logger import get_module_logger

logger = get_module_logger("easy_organizer")


class EasyOrganizer(BaseOrganizer):
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
        self.use_flags = {}
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
        (
            self.learnware_list,
            self.learnware_zip_list,
            self.learnware_folder_list,
            self.use_flags,
            self.count,
        ) = self.dbops.load_market()

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, id: str = None, check_status: int = None
    ) -> Tuple[str, bool]:
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
        logger.info("Get new learnware from %s" % (zip_path))

        id = id if id is not None else "%08d" % (self.count)
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
            return None, BaseChecker.INVALID_LEARNWARE

        if new_learnware is None:
            return None, BaseChecker.INVALID_LEARNWARE

        learnwere_status = check_status if check_status is not None else BaseChecker.NONUSABLE_LEARNWARE

        self.dbops.add_learnware(
            id=id,
            semantic_spec=semantic_spec,
            zip_path=target_zip_dir,
            folder_path=target_folder_dir,
            use_flag=learnwere_status,
        )

        self.learnware_list[id] = new_learnware
        self.learnware_zip_list[id] = target_zip_dir
        self.learnware_folder_list[id] = target_folder_dir
        self.use_flags[id] = learnwere_status
        self.count += 1
        return id, learnwere_status

    def delete_learnware(self, id: str) -> bool:
        """Delete Learnware from market

        Parameters
        ----------
        id : str
            Learnware to be deleted

        Returns
        -------
        bool
            True for successful operation.
            False for id not found.
        """
        if not id in self.learnware_list:
            logger.warning("Learnware id:'{}' NOT Found!".format(id))
            return False

        zip_dir = self.learnware_zip_list[id]
        os.remove(zip_dir)
        folder_dir = self.learnware_folder_list[id]
        rmtree(folder_dir)
        self.learnware_list.pop(id)
        self.learnware_zip_list.pop(id)
        self.learnware_folder_list.pop(id)
        self.use_flags.pop(id)
        self.dbops.delete_learnware(id=id)

        return True

    def update_learnware(self, id: str, zip_path: str = None, semantic_spec: dict = None, check_status: int = None):
        """update learnware with zip_path and semantic_specification
        TODO: update should pass the semantic check too

        Parameters
        ----------
        id : str
            _description_
        zip_path : str, optional
            _description_, by default None
        semantic_spec : dict, optional
            _description_, by default None
        check_status : int, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        assert (
            zip_path is None and semantic_spec is None
        ), f"at least one of 'zip_path' and 'semantic_spec' should not be None when update learnware"
        assert check_status != BaseChecker.INVALID_LEARNWARE, f"'check_status' can not be INVALID_LEARNWARE"

        if zip_path is None and check_status is not None:
            logger.warning("check_status will be ignored when zip_path is None for learnware update")

        learnware_zippath = self.learnware_zip_list[id] if zip_path is None else zip_path
        semantic_spec = (
            self.learnware_list[id].get_specification().get_semantic_spec() if semantic_spec is None else semantic_spec
        )

        self.dbops.update_learnware_semantic_specification(id, semantic_spec)

        target_zip_dir = self.learnware_zip_list[id]
        target_folder_dir = self.learnware_folder_list[id]

        if check_status is None and zip_path is not None:
            with tempfile.TemporaryDirectory(prefix="learnware_") as tempdir:
                with zipfile.ZipFile(zip_path, "r") as z_file:
                    z_file.extractall(tempdir)

                try:
                    new_learnware = get_learnware_from_dirpath(
                        id=id, semantic_spec=semantic_spec, learnware_dirpath=tempdir
                    )
                except Exception:
                    return BaseChecker.INVALID_LEARNWARE

                if new_learnware is None:
                    return BaseChecker.INVALID_LEARNWARE

                learnwere_status = BaseChecker.NONUSABLE_LEARNWARE
        else:
            learnwere_status = self.use_flags[id] if zip_path is None else check_status

        copyfile(zip_path, target_zip_dir)
        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)

        self.learnware_list[id] = get_learnware_from_dirpath(
            id=id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
        )
        self.use_flags[id] = learnwere_status
        self.dbops.update_learnware_use_flag(id, learnwere_status)
        return learnwere_status

    def get_learnware_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """Search learnware by id or list of ids.

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of targer learware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return target learnware or list of target learnwares.
            None for Learnware NOT Found.
        """
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_list:
                    ret.append(self.learnware_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

    def get_learnware_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
        """Get Zipped Learnware file by id

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of targer learware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        Union[Learnware, List[Learnware]]
            Return the path for target learnware or list of path.
            None for Learnware NOT Found.
        """
        if isinstance(ids, list):
            ret = []
            for id in ids:
                if id in self.learnware_zip_list:
                    ret.append(self.learnware_zip_list[id])
                else:
                    logger.warning("Learnware ID '%s' NOT Found!" % (id))
                    ret.append(None)
            return ret
        else:
            try:
                return self.learnware_zip_list[ids]
            except:
                logger.warning("Learnware ID '%s' NOT Found!" % (ids))
                return None

    def get_learnware_ids(self, top: int = None) -> List[str]:
        if top is None:
            return list(self.learnware_list.keys())
        else:
            return list(self.learnware_list.keys())[:top]

    def get_learnwares(self, top: int = None) -> List[str]:
        if top is None:
            return list(self.learnware_list.values())
        else:
            return list(self.learnware_list.values())[:top]

    def __len__(self):
        return len(self.learnware_list)
