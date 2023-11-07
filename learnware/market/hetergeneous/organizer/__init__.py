from __future__ import annotations

import copy
import multiprocessing
import os
import tempfile
import zipfile
from collections import defaultdict
from shutil import copyfile, rmtree
from typing import List

import pandas as pd
from torch import nn
import torch.multiprocessing as mp

from ....learnware import Learnware, get_learnware_from_dirpath
from ....logger import get_module_logger
from ....specification.system import HeteroSpecification
from ...base import BaseChecker, BaseUserInfo
from ...easy2 import EasyOrganizer
from ..database_ops import DatabaseOperations
from .config import C as conf
from .hetero_mapping import HeteroMapping, Trainer

logger = get_module_logger("hetero_market")


class HeteroMapTableOrganizer(EasyOrganizer):
    def reload_market(self, rebuild=False, auto_update_limit=100):
        self.market_store_path = os.path.join(conf.market_root_path, self.market_id)
        self.market_mapping_path = os.path.join(self.market_store_path, conf.market_model_path)
        self.learnware_pool_path = os.path.join(self.market_store_path, "learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.market_store_path, "zips")
        self.learnware_folder_pool_path = os.path.join(self.market_store_path, "unzipped_learnwares")
        self.learnware_list = {}  # id:learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.count = 0
        self.last_trained_learnware_num = 0
        self.dbops = DatabaseOperations(conf.database_url, "market_" + self.market_id)
        self.auto_update = False
        self.auto_update_limit = auto_update_limit
        self.auto_update_lock = mp.Lock()
        self.is_training_in_progress = mp.Value('i', 0)

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
        # model training arguments(model architecture + optimization) set via self.reset
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

        with self.auto_update_lock:
            if self.auto_update and not self.is_training_in_progress.value and self.count - self.last_trained_learnware_num >= self.auto_update_limit:
                self.is_training_in_progress.value = 1
                curr_learnware_list = copy.deepcopy(self.learnware_list)
                train_process = mp.Process(target=self.train, args=(curr_learnware_list.values(),))
                train_process.start()
                # train_process.join()
        
        return learnware_id, learnwere_status

    def train(self, learnware_list: List[Learnware] = None):
        learnware_list = learnware_list or self.learnware_list.values()
        logger.warning(f"Leanwares for training: {[learnware.id for learnware in learnware_list]}")
        allset = self._learnwares_to_dataframes(learnware_list)
        self.market_mapping = HeteroMapping(**self.training_args)
        market_mapping_trainer = Trainer(
            model=self.market_mapping,
            train_set_list=allset,
            collate_fn=self.market_mapping.collate_fn,
            **self.training_args,
        )
        market_mapping_trainer.train()

        # auto save whenever market model retrained
        market_mapping_trainer.save_model(output_dir=self.market_store_path)

        # essential hetero-mapping update for each market learnware when market model retrained
        self._update_learnware_list(learnware_list)
        self.last_trained_learnware_num = self.count

        logger.warning(f"Updataed Specification For: {[learnware.id for learnware in learnware_list]}")

        with self.auto_update_lock:
            self.is_training_in_progress.value = 0

    def _update_learnware_list(self, learnware_list: List[Learnware]):
        try:
            hetero_mappings_save_path = os.path.join(self.market_store_path, "hetero_mappings")
            os.makedirs(hetero_mappings_save_path, exist_ok=True)
            for learnware in learnware_list:
                hetero_spec_path = os.path.join(hetero_mappings_save_path, f"{learnware.id}.npy")
                self._update_learnware_specification(learnware, save_path=hetero_spec_path)
                logger.info(f"Learnware {learnware.id} HeteroSpecification Successfully Saved")
        except Exception as err:
            logger.warning(f"Update learnware HeteroSpecification failed! Due to {err}")

    def _update_learnware_specification(self, learnware: Learnware, save_path: str) -> Learnware:
        specification = learnware.specification
        learnware_rkme = specification.get_stat_spec()["RKMETableSpecification"]
        learnware_features = specification.get_semantic_spec()["Input"]["Description"].values()
        learnware_hetero_spec = self.market_mapping.hetero_mapping(learnware_rkme, learnware_features)
        learnware.update_stat_spec("HeteroSpecification", learnware_hetero_spec)

        # custom hetero spec save path?
        learnware_hetero_spec.save(save_path)

    def generate_hetero_map_spec(self, user_info: BaseUserInfo) -> HeteroSpecification:
        user_rkme = user_info.stat_info["RKMETableSpecification"]
        user_features = user_info.semantic_spec["Input"]["Description"].values()
        user_hetero_spec = self.market_mapping.hetero_mapping(user_rkme, user_features)
        return user_hetero_spec

    def _learnwares_to_dataframes(self, learnware_list: List[Learnware]) -> List[pd.DataFrame]:
        learnware_df_dict = defaultdict(list)
        for learnware in learnware_list:
            specification = learnware.get_specification()
            learnware_rkme = specification.get_stat_spec()["RKMETableSpecification"]
            learnware_features = specification.get_semantic_spec()["Input"]["Description"]
            learnware_df = pd.DataFrame(data=learnware_rkme.get_z(), columns=learnware_features.values())

            learnware_df_dict[tuple(sorted(learnware_features))].append(learnware_df)

        merged_dfs = [pd.concat(dfs) for dfs in learnware_df_dict.values()]
        return merged_dfs

    def save(self, save_path):
        return NotImplementedError