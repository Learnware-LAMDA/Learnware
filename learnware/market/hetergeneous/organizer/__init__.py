from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from typing import List

import pandas as pd

from ....learnware import Learnware
from ....logger import get_module_logger
from ....specification.system import HeteroSpecification
from ...base import BaseOrganizer, BaseUserInfo
from ..database_ops import DatabaseOperations
from .config import C as conf
from .hetero_mapping import HeteroMapping, Trainer

logger = get_module_logger("hetero_market")


class HeteroMapTableOrganizer(BaseOrganizer):
    def reload_market(self, rebuild=False, auto_update_limit=50):
        self.market_store_path = os.path.join(conf.market_root_path, self.market_id)
        self.market_mapping_path = os.path.join(self.market_store_path, conf.market_model_path)
        self.learnware_pool_path = os.path.join(self.market_store_path, "learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.market_store_path, "zips")
        self.learnware_folder_pool_path = os.path.join(self.market_store_path, "unzipped_learnwares")
        self.learnware_list = {}  # id:learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.count = 0
        # default root path: ../../.learnware
        self.root_path = conf.market_root_path
        self.dbops = DatabaseOperations(conf.database_url, "market_" + self.market_id)
        self.auto_update_limit = auto_update_limit

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

        if rebuild:
            logger.warning("Warning! You are trying to clear current database!")
            try:
                self.dbops.clear_learnware_table()
                rmtree(self.learnware_pool_path)
            except:
                pass
        else:
            if os.path.exists(self.market_mapping_path):
                logger.info(f"Loading Market Mapping from Default Checkpoint {self.market_mapping_path}")
                self.market_mapping = HeteroMapping.load(checkpoint=self.market_store_path)
                # self._update_learnware_list(self.learnware_list)
            else:
                logger.warning(f"No Existing Market Mapping!!")
                self.market_mapping = HeteroMapping()

    def reset(self, market_id=None, auto_update=False, **kwargs):
        # model training arguments(model architecture + optimization) set via self.reset
        self.auto_update = auto_update
        self.market_id = market_id
        self.training_args = kwargs

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, check_status: int, learnware: Learnware
    ) -> Tuple[str, int]:
        self._update_learnware_list([learnware])
        self.learnware_list[learnware.id] = learnware
        self.count += 1

        if self.auto_update and self.count >= self.auto_update_limit:
            train_process = multiprocessing.Process(target=self.train, args=(self.learnware_list,))
            train_process.start()
            # train_process.join()

    def delete_learnware(self, id: str) -> bool:
        raise NotImplementedError

    def update_learnware(self, learnware: Learnware):
        raise NotImplementedError

    def get_learnwares(self):
        return self.learnware_list

    def train(self, learnware_list: List[Learnware]):
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

    def _update_learnware_list(self, learnware_list: List[Learnware]):
        hetero_mappings_save_path = os.path.join(self.market_store_path, "hetero_mappings")
        os.makedirs(hetero_mappings_save_path, exist_ok=True)
        for learnware in learnware_list:
            learnware.id = learnware.id.replace(",", "_")
            hetero_spec_path = os.path.join(hetero_mappings_save_path, f"{learnware.id}.npy")
            self._update_learnware_specification(learnware, save_path=hetero_spec_path)

    def _update_learnware_specification(self, learnware: Learnware, save_path: str) -> Learnware:
        specification = learnware.specification
        learnware_rkme = specification.get_stat_spec()["RKMEStatSpecification"]
        learnware_features = specification.get_semantic_spec()["Input"]["Description"].values()
        learnware_hetero_spec = self.market_mapping.hetero_mapping(learnware_rkme, learnware_features)
        learnware.update_stat_spec("HeteroSpecification", learnware_hetero_spec)

        # custom hetero spec save path?
        learnware_hetero_spec.save(save_path)

    def generate_hetero_map_spec(self, user_info: BaseUserInfo) -> HeteroSpecification:
        user_rkme = user_info.stat_info["RKMEStatSpecification"]
        user_features = user_info.semantic_spec["Input"]["Description"].values()
        user_hetero_spec = self.market_mapping.hetero_mapping(user_rkme, user_features)
        return user_hetero_spec

    def _learnwares_to_dataframes(self, learnware_list: List[Learnware]) -> List[pd.DataFrame]:
        learnware_df_dict = defaultdict(list)
        for learnware in learnware_list:
            specification = learnware.get_specification()
            learnware_rkme = specification.get_stat_spec()["RKMEStatSpecification"]
            learnware_features = specification.get_semantic_spec()["Input"]["Description"]
            learnware_df = pd.DataFrame(data=learnware_rkme.get_z(), columns=learnware_features.values())

            learnware_df_dict[tuple(sorted(learnware_features))].append(learnware_df)

        merged_dfs = [pd.concat(dfs) for dfs in learnware_df_dict.values()]
        return merged_dfs

    def save(self, save_path):
        return NotImplementedError

    def __len__(self):
        return len(self.learnware_list)
