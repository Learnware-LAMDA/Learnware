from __future__ import annotations

import copy
import multiprocessing
import os
import tempfile
import zipfile
from collections import defaultdict
from shutil import copyfile, rmtree
from typing import List, Tuple

import pandas as pd
from torch import nn
import torch.multiprocessing as mp

from ....learnware import Learnware, get_learnware_from_dirpath
from ....logger import get_module_logger
from ....specification.system import HeteroSpecification
from ...base import BaseChecker, BaseUserInfo
from ...easy import EasyOrganizer
from ...easy.database_ops import DatabaseOperations
from ....config import C as conf
from .hetero_mapping import HeteroMapping, Trainer

logger = get_module_logger("hetero_market")


class HeteroMapTableOrganizer(EasyOrganizer):
    def reload_market(self, rebuild=False, auto_update_limit=100):
        self.market_store_path = os.path.join(conf.root_path, self.market_id)
        self.market_mapping_path = os.path.join(self.market_store_path, "model.bin")
        self.learnware_pool_path = os.path.join(self.market_store_path, "learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.market_store_path, "zips")
        self.learnware_folder_pool_path = os.path.join(self.market_store_path, "unzipped_learnwares")
        self.hetero_mappings_path = os.path.join(self.market_store_path, "hetero_mappings")
        self.learnware_list = {}  # id:learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.count = 0
        self.last_trained_learnware_num = 0
        self.dbops = DatabaseOperations(conf.database_url, "market_" + self.market_id)
        self.auto_update = False
        self.auto_update_limit = auto_update_limit

        if rebuild:
            logger.warning("Warning! You are trying to clear current database!")
            try:
                self.dbops.clear_learnware_table()
                rmtree(self.learnware_pool_path)
            except Exception as err:
                logger.warning(f"Clear current database failed due to {err}!!")
                pass

        os.makedirs(self.learnware_pool_path, exist_ok=True)
        os.makedirs(self.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(self.learnware_folder_pool_path, exist_ok=True)
        os.makedirs(self.hetero_mappings_path, exist_ok=True)

        (
            self.learnware_list,
            self.learnware_zip_list,
            self.learnware_folder_list,
            self.use_flags,
            self.count,
        ) = self.dbops.load_market()

        if os.path.exists(self.market_mapping_path):
            logger.info(f"Loading Market Mapping from Default Checkpoint {self.market_mapping_path}")
            self.market_mapping = HeteroMapping.load(checkpoint=self.market_store_path)
            # self._update_learnware_list(self.learnware_list)
        else:
            logger.warning(f"No Existing Market Mapping!!")
            self.market_mapping = HeteroMapping()

    def reset(self, market_id=None, auto_update=False, auto_update_limit=None, **kwargs):
        self.auto_update = auto_update
        self.market_id = market_id
        self.training_args = kwargs
        if auto_update_limit is not None: self.auto_update_limit = auto_update_limit

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, check_status: int, learnware_id: str = None
    ) -> Tuple[str, int]:
        logger.info("Get new learnware from %s" % (zip_path))

        learnware_id = "%08d" % (self.count) if learnware_id is None else learnware_id
        target_zip_dir = os.path.join(self.learnware_zip_pool_path, "%s.zip" % (learnware_id))
        target_folder_dir = os.path.join(self.learnware_folder_pool_path, learnware_id)
        copyfile(zip_path, target_zip_dir)

        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        logger.info("Learnware move to %s, and unzip to %s" % (target_zip_dir, target_folder_dir))

        try:
            new_learnware = get_learnware_from_dirpath(
                id=learnware_id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
            )
        except:
            logger.info("New Learnware Not Properly Added!!!")
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
            id=learnware_id,
            semantic_spec=semantic_spec,
            zip_path=target_zip_dir,
            folder_path=target_folder_dir,
            use_flag=learnwere_status,
        )

        self._update_learnware_list([new_learnware])
        self.learnware_list[learnware_id] = new_learnware
        self.learnware_zip_list[learnware_id] = target_zip_dir
        self.learnware_folder_list[learnware_id] = target_folder_dir
        self.use_flags[learnware_id] = learnwere_status
        self.count += 1

        if self.auto_update and self.count - self.last_trained_learnware_num == self.auto_update_limit + 1:
            logger.warning(f"Leanwares for training: {self.get_learnware_ids()}")

            updated_market_mapping = self.train(
                learnware_list=self.learnware_list.values(),
                save_dir=self.market_store_path, 
                **self.training_args
            )
            
            logger.warning(f"Market mapping train completed. Now update HeteroSpecification for {self.get_learnware_ids()}")
            
            self.market_mapping = updated_market_mapping
            self._update_learnware_list(self.learnware_list.values())
            self.last_trained_learnware_num = self.count
        
        return learnware_id, learnwere_status

    @staticmethod
    def train(learnware_list: List[Learnware] = None, save_dir: str = None, **kwargs):
        allset = HeteroMapTableOrganizer._learnwares_to_dataframes(learnware_list)
        market_mapping = HeteroMapping(**kwargs)
        market_mapping_trainer = Trainer(
            model=market_mapping,
            train_set_list=allset,
            collate_fn=market_mapping.collate_fn,
            **kwargs,
        )

        market_mapping_trainer.train()
        market_mapping_trainer.save_model(output_dir=save_dir)

        return market_mapping

        ############################################
        # save_model & generateing new specification
        # should be moved out of train thread
        ############################################

    def _update_learnware_list(self, learnware_list: List[Learnware]):
        try:
            for learnware in learnware_list:
                hetero_spec_path = os.path.join(self.hetero_mappings_path, f"{learnware.id}.npy")
                self._update_learnware_specification(learnware, save_path=hetero_spec_path)
        except Exception as err:
            logger.warning(f"Update HeteroSpecification failed! Due to {err}")

    def _update_learnware_specification(self, learnware: Learnware, save_path: str) -> Learnware:
        specification = learnware.specification
        learnware_rkme = specification.get_stat_spec()["RKMETableSpecification"]
        learnware_features = specification.get_semantic_spec()["Input"]["Description"].values()
        learnware_hetero_spec = self.market_mapping.hetero_mapping(learnware_rkme, learnware_features)
        learnware.update_stat_spec("HeteroSpecification", learnware_hetero_spec)

        learnware_hetero_spec.save(save_path)

    def generate_hetero_map_spec(self, user_info: BaseUserInfo) -> HeteroSpecification:
        user_rkme = user_info.stat_info["RKMETableSpecification"]
        user_features = user_info.get_semantic_spec()["Input"]["Description"].values()
        user_hetero_spec = self.market_mapping.hetero_mapping(user_rkme, user_features)
        return user_hetero_spec

    @staticmethod
    def _learnwares_to_dataframes(learnware_list: List[Learnware]) -> List[pd.DataFrame]:
        learnware_df_dict = defaultdict(list)
        for learnware in learnware_list:
            specification = learnware.get_specification()
            learnware_rkme = specification.get_stat_spec()["RKMETableSpecification"]
            learnware_features = specification.get_semantic_spec()["Input"]["Description"]
            learnware_df = pd.DataFrame(data=learnware_rkme.get_z(), columns=learnware_features.values())

            learnware_df_dict[tuple(sorted(learnware_features))].append(learnware_df)

        merged_dfs = [pd.concat(dfs) for dfs in learnware_df_dict.values()]
        return merged_dfs