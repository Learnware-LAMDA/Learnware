import os
import copy
import zipfile
import pandas as pd
from collections import defaultdict
from shutil import copyfile, rmtree
from typing import List, Tuple

from ....learnware import Learnware, get_learnware_from_dirpath
from ....logger import get_module_logger
from ....specification.system import HeteroMapTableSpecification
from ...base import BaseChecker, BaseUserInfo
from ...easy import EasyOrganizer
from ...easy.database_ops import DatabaseOperations
from ....config import C as conf
from .hetero_map import HeteroMap, Trainer

logger = get_module_logger("hetero_map_table_organizer")


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
        self.training_count = 1
        self.last_training_count = 0
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
            logger.info(f"Reload market mapping from checkpoint {self.market_mapping_path}")
            self.market_mapping = HeteroMap.load(checkpoint=self.market_store_path)
            if not rebuild:
                if os.path.exists(self.hetero_mappings_path):
                    for hetero_json_path in os.listdir(self.hetero_mappings_path):
                        idx = hetero_json_path.split(".")[0]
                        hetero_spec = HeteroMapTableSpecification()
                        hetero_spec.load(os.path.join(self.hetero_mappings_path, f"{idx}.json"))
                        try:
                            self.learnware_list[idx].update_stat_spec("HeteroMapTableSpecification", hetero_spec)
                        except:
                            logger.warning(f"Learnware ID {idx} NOT Found!")
                else:
                    logger.info("No HeteroMapTableSpecification to reload. Use loaded market mapping to regenerate.")
                    self._update_learnware_by_ids(self.learnware_list.keys())
        else:
            logger.warning(f"No market mapping to reload!!")
            self.market_mapping = HeteroMap()
            # rmtree(self.hetero_mappings_path)

    def reset(self, market_id=None, auto_update=False, auto_update_limit=None, **kwargs):
        self.auto_update = auto_update
        self.market_id = market_id
        self.training_args = kwargs
        if auto_update_limit is not None:
            self.auto_update_limit = auto_update_limit

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, check_status: int, learnware_id: str = None
    ) -> Tuple[str, int]:
        if check_status == BaseChecker.INVALID_LEARNWARE:
            logger.warning("Learnware is invalid!")
            return None, BaseChecker.INVALID_LEARNWARE

        semantic_spec = copy.deepcopy(semantic_spec)
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

        self.learnware_list[learnware_id] = new_learnware
        self.learnware_zip_list[learnware_id] = target_zip_dir
        self.learnware_folder_list[learnware_id] = target_folder_dir
        self.use_flags[learnware_id] = learnwere_status
        self._update_learnware_by_ids([learnware_id])
        self.count += 1
        self.training_count += [learnware_id] == self._get_table_type_learnware_ids([learnware_id])

        if self.auto_update and self.training_count - self.last_training_count == self.auto_update_limit + 1:
            training_learnware_ids = self._get_table_type_learnware_ids(self.get_learnware_ids())
            training_learnwares = self.get_learnware_by_ids(training_learnware_ids)
            logger.warning(f"Leanwares for training: {training_learnware_ids}")

            updated_market_mapping = self.train(
                learnware_list=training_learnwares, save_dir=self.market_store_path, **self.training_args
            )

            logger.warning(
                f"Market mapping train completed. Now update HeteroMapTableSpecification for {training_learnware_ids}"
            )
            self.market_mapping = updated_market_mapping
            self._update_learnware_by_ids(training_learnware_ids)
            self.last_training_count = len(training_learnware_ids)

        return learnware_id, learnwere_status

    @staticmethod
    def train(learnware_list: List[Learnware], save_dir: str, **kwargs):
        allset = HeteroMapTableOrganizer._learnwares_to_dataframes(learnware_list)
        market_mapping = HeteroMap(**kwargs)
        market_mapping_trainer = Trainer(
            model=market_mapping,
            train_set_list=allset,
            collate_fn=market_mapping.collate_fn,
            **kwargs,
        )

        market_mapping_trainer.train()
        market_mapping_trainer.save_model(output_dir=save_dir)

        return market_mapping

    def _update_learnware_by_ids(self, ids: List[str]):
        ids = self._get_table_type_learnware_ids(ids)
        for id in ids:
            try:
                spec = self.learnware_list[id].get_specification()
                semantic_spec, stat_spec = spec.get_semantic_spec(), spec.get_stat_spec()["RKMETableSpecification"]
                features = semantic_spec["Input"]["Description"].values()
                hetero_spec = self.market_mapping.hetero_mapping(stat_spec, features)
                self.learnware_list[id].update_stat_spec("HeteroMapTableSpecification", hetero_spec)

                save_path = os.path.join(self.hetero_mappings_path, f"{id}.json")
                hetero_spec.save(save_path)
            except Exception as err:
                logger.warning(f"Learnware {id} generate HeteroMapTableSpecification failed! Due to {err}")

    def generate_hetero_map_spec(self, user_info: BaseUserInfo) -> HeteroMapTableSpecification:
        user_stat_spec = user_info.stat_info["RKMETableSpecification"]
        user_features = user_info.get_semantic_spec()["Input"]["Description"].values()

        user_hetero_spec = self.market_mapping.hetero_mapping(user_stat_spec, user_features)
        return user_hetero_spec

    @staticmethod
    def _learnwares_to_dataframes(learnware_list: List[Learnware]) -> List[pd.DataFrame]:
        learnware_df_dict = defaultdict(list)
        for learnware in learnware_list:
            spec = learnware.get_specification()
            stat_spec = spec.get_stat_spec()["RKMETableSpecification"]
            features = spec.get_semantic_spec()["Input"]["Description"]
            learnware_df = pd.DataFrame(data=stat_spec.get_z(), columns=features.values())
            learnware_df_dict[tuple(sorted(features))].append(learnware_df)

        return [pd.concat(dfs) for dfs in learnware_df_dict.values()]

    def _get_table_type_learnware_ids(self, ids: List[str]) -> List[str]:
        ret = []
        for id in ids:
            semantic_spec = self.learnware_list[id].get_specification().get_semantic_spec()
            if semantic_spec["Data"]["Values"][0] == "Table":
                ret.append(id)
        return ret
