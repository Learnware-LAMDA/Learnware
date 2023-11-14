import os
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Union

from ....learnware import Learnware
from ....logger import get_module_logger
from ....specification import RKMETableSpecification, HeteroMapTableSpecification
from ...base import BaseChecker, BaseUserInfo
from ...easy import EasyOrganizer
from .hetero_map import HeteroMap, Trainer

logger = get_module_logger("hetero_map_table_organizer")


class HeteroMapTableOrganizer(EasyOrganizer):
    def reload_market(self, rebuild=False):
        super(HeteroMapTableOrganizer, self).reload_market(rebuild=rebuild)

        hetero_folder_path = os.path.join(self.market_store_path, "hetero")
        os.makedirs(hetero_folder_path, exist_ok=True)
        self.market_mapping_path = os.path.join(hetero_folder_path, "model.bin")
        self.hetero_specs_path = os.path.join(hetero_folder_path, "hetero_specifications")
        self.training_args = {"cache_dir": hetero_folder_path}
        os.makedirs(self.hetero_specs_path, exist_ok=True)

        if os.path.exists(self.market_mapping_path):
            logger.info(f"Reload market mapping from checkpoint {self.market_mapping_path}")
            self.market_mapping = HeteroMap.load(checkpoint=self.market_mapping_path)
            if not rebuild:
                if os.path.exists(self.hetero_specs_path):
                    for hetero_json_path in os.listdir(self.hetero_specs_path):
                        try:
                            idx = hetero_json_path.split(".")[0]
                            hetero_spec = HeteroMapTableSpecification()
                            hetero_spec.load(os.path.join(self.hetero_specs_path, f"{idx}.json"))
                            self.learnware_list[idx].update_stat_spec(hetero_spec.type, hetero_spec)
                        except:
                            logger.warning(f"Learnware {idx} hetero spec loaded failed!")
                else:
                    logger.info("No HeteroMapTableSpecification to reload. Use loaded market mapping to regenerate.")
                    self._update_learnware_by_ids(self.get_learnware_ids(check_status=BaseChecker.USABLE_LEARWARE))
        else:
            logger.warning(f"No market mapping to reload!")
            self.market_mapping = HeteroMap(cache_dir=hetero_folder_path)

    def reset(self, market_id, rebuild=False, auto_update=False, auto_update_limit=100, **training_args):
        super(HeteroMapTableOrganizer, self).reset(market_id, rebuild)
        self.auto_update = auto_update
        self.auto_update_limit = auto_update_limit
        self.count_down = auto_update_limit
        self.training_args = training_args

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, check_status: int, learnware_id: str = None
    ) -> Tuple[str, int]:
        learnware_id, learnwere_status = super(HeteroMapTableOrganizer, self).add_learnware(
            zip_path, semantic_spec, check_status, learnware_id
        )

        if learnwere_status == BaseChecker.USABLE_LEARWARE and len(self._get_hetero_learnware_ids(learnware_id)):
            self._update_learnware_by_ids(learnware_id)

            if self.auto_update:
                self.count_down -= 1
                if self.count_down == 0:
                    training_learnware_ids = self._get_hetero_learnware_ids(
                        self.get_learnware_ids(check_status=BaseChecker.USABLE_LEARWARE)
                    )
                    training_learnwares = self.get_learnware_by_ids(training_learnware_ids)
                    logger.info(f"Verified leanwares for training: {training_learnware_ids}")
                    updated_market_mapping = self.train(
                        learnware_list=training_learnwares, save_dir=self.market_mapping_path, **self.training_args
                    )
                    logger.info(
                        f"Market mapping train completed. Now update HeteroMapTableSpecification for {training_learnware_ids}"
                    )
                    self.market_mapping = updated_market_mapping
                    self._update_learnware_by_ids(training_learnware_ids)

                    self.count_down = self.auto_update_limit

        return learnware_id, learnwere_status

    def delete_learnware(self, id: str) -> bool:
        flag = super(HeteroMapTableOrganizer, self).delete_learnware(id)
        if flag:
            hetero_spec_path = os.path.join(self.hetero_specs_path, f"{id}.json")
            try:
                os.remove(hetero_spec_path)
            except FileNotFoundError:
                pass
        return flag

    def update_learnware(self, id: str, zip_path: str = None, semantic_spec: dict = None, check_status: int = None):
        final_status = super(HeteroMapTableOrganizer, self).update_learnware(id, zip_path, semantic_spec, check_status)
        if final_status == BaseChecker.USABLE_LEARWARE and len(self._get_hetero_learnware_ids(id)):
            self._update_learnware_by_ids(id)
        return final_status

    def reload_learnware(self, learnware_id: str):
        super(HeteroMapTableOrganizer, self).reload_learnware(learnware_id)
        try:
            hetero_spec_path = os.path.join(self.hetero_specs_path, f"{learnware_id}.json")
            if os.path.exists(hetero_spec_path):
                hetero_spec = HeteroMapTableSpecification()
                hetero_spec.load(hetero_spec_path)
                self.learnware_list[learnware_id].update_stat_spec(hetero_spec.type, hetero_spec)
        except:
            logger.warning(f"Learnware {learnware_id} hetero spec loaded failed!")

    def _update_learnware_by_ids(self, ids: Union[str, List[str]]):
        ids = self._get_hetero_learnware_ids(ids)
        for idx in ids:
            try:
                spec = self.learnware_list[idx].get_specification()
                semantic_spec, stat_spec = spec.get_semantic_spec(), spec.get_stat_spec()["RKMETableSpecification"]
                features = semantic_spec["Input"]["Description"]
                save_path = os.path.join(self.hetero_specs_path, f"{idx}.json")

                hetero_spec = self.market_mapping.hetero_mapping(stat_spec, features)
                self.learnware_list[idx].update_stat_spec(hetero_spec.type, hetero_spec)
                hetero_spec.save(save_path)

            except Exception as err:
                logger.warning(f"Learnware {idx} generate HeteroMapTableSpecification failed! Due to {err}")

    def _get_hetero_learnware_ids(self, ids: Union[str, List[str]]) -> List[str]:
        if isinstance(ids, str):
            ids = [ids]

        ret = []
        for idx in ids:
            try:
                spec = self.learnware_list[idx].get_specification()
                semantic_spec, rkme = spec.get_semantic_spec(), spec.get_stat_spec().get("RKMETableSpecification", None)
                if isinstance(rkme, RKMETableSpecification) and isinstance(semantic_spec["Input"], dict):
                    ret.append(idx)
            except:
                continue
        return ret

    def generate_hetero_map_spec(self, user_info: BaseUserInfo) -> HeteroMapTableSpecification:
        user_stat_spec = user_info.stat_info["RKMETableSpecification"]
        user_features = user_info.get_semantic_spec()["Input"]["Description"]
        user_hetero_spec = self.market_mapping.hetero_mapping(user_stat_spec, user_features)
        return user_hetero_spec

    @staticmethod
    def train(learnware_list: List[Learnware], save_dir: str, **kwargs) -> HeteroMap:
        # Convert learnware to dataframe
        learnware_df_dict = defaultdict(list)
        for learnware in learnware_list:
            spec = learnware.get_specification()
            stat_spec = spec.get_stat_spec()["RKMETableSpecification"]
            features = spec.get_semantic_spec()["Input"]["Description"]
            learnware_df = pd.DataFrame(data=stat_spec.get_z(), columns=features.values())
            learnware_df_dict[tuple(sorted(features))].append(learnware_df)
        allset = [pd.concat(dfs) for dfs in learnware_df_dict.values()]

        # Train market mapping
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
