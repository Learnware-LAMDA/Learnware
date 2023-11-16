import os
import traceback
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Union

import pandas as pd

from .hetero_map import HeteroMap, Trainer
from ..utils import is_hetero
from ...base import BaseChecker, BaseUserInfo
from ...easy import EasyOrganizer
from ....learnware import Learnware
from ....logger import get_module_logger
from ....specification import HeteroMapTableSpecification, RKMETableSpecification


logger = get_module_logger("hetero_map_table_organizer")


class HeteroMapTableOrganizer(EasyOrganizer):
    def reload_market(self, rebuild=False) -> bool:
        """Reload the heterogeneous learnware organizer when server restarted.

        Returns
        -------
        bool
            A flag indicating whether the heterogeneous market is reloaded successfully.
        """
        super(HeteroMapTableOrganizer, self).reload_market(rebuild=rebuild)

        hetero_folder_path = os.path.join(self.market_store_path, "hetero")
        os.makedirs(hetero_folder_path, exist_ok=True)
        self.market_mapping_path = os.path.join(hetero_folder_path, "model.bin")
        self.hetero_specs_path = os.path.join(hetero_folder_path, "hetero_specifications")
        os.makedirs(self.hetero_specs_path, exist_ok=True)

        if os.path.exists(self.market_mapping_path):
            logger.info(f"Reload market mapping from checkpoint {self.market_mapping_path}")
            self.market_mapping = HeteroMap.load(checkpoint=self.market_mapping_path)
            if not rebuild:
                usable_ids = self.get_learnware_ids(check_status=BaseChecker.USABLE_LEARWARE)
                hetero_ids = self._get_hetero_learnware_ids(usable_ids)
                for hetero_id in hetero_ids:
                    self._reload_learnware_hetero_spec(hetero_id)
        else:
            logger.warning(f"No market mapping to reload!")
            self.market_mapping = HeteroMap()

    def reset(self, market_id, rebuild=False, auto_update=False, auto_update_limit=100, **training_args):
        """Reset the heterogeneous market with specified settings.

        Parameters
        ----------
        market_id : str
            the heterogeneous market's id
        rebuild : bool, optional
            A flag indicating whether to reload market, by default False
        auto_update : bool, optional
            A flag indicating whether to enable automatic updating of market mapping, by default False
        auto_update_limit : int, optional
            The threshold for the number of learnwares required to trigger an automatic market mapping update, by default 100
        """
        self.auto_update = auto_update
        self.auto_update_limit = auto_update_limit
        self.count_down = auto_update_limit
        self.training_args = training_args

        super(HeteroMapTableOrganizer, self).reset(market_id, rebuild)

    def add_learnware(
        self, zip_path: str, semantic_spec: dict, check_status: int, learnware_id: str = None
    ) -> Tuple[str, int]:
        """Add a learnware into the heterogeneous learnware market.
           Initiates an update of the market mapping if `auto_update` is True and the number of learnwares supporting training reaches `auto_update_limit`.

        Parameters
        ----------
        zip_path : str
            Filepath for learnware model, a zipped file.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.
        check_status : int
            A flag indicating whether the learnware is usable.
        learnware_id : str, optional
            A id in database for learnware

        Returns
        -------
        Tuple[str, int]
            - str indicating model_id
            - int indicating the final learnware check_status
        """
        learnware_id, learnwere_status = super(HeteroMapTableOrganizer, self).add_learnware(
            zip_path, semantic_spec, check_status, learnware_id
        )

        if learnwere_status == BaseChecker.USABLE_LEARWARE and len(self._get_hetero_learnware_ids(learnware_id)):
            self._update_learware_hetero_sepc(learnware_id)

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
                    self._update_learware_hetero_sepc(training_learnware_ids)

                    self.count_down = self.auto_update_limit

        return learnware_id, learnwere_status

    def delete_learnware(self, id: str) -> bool:
        """Delete learnware from heterogeneous learnware market.
           If a corresponding HeteroMapTableSpecification exists, it is also removed.

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
        flag = super(HeteroMapTableOrganizer, self).delete_learnware(id)
        if flag:
            hetero_spec_path = os.path.join(self.hetero_specs_path, f"{id}.json")
            try:
                os.remove(hetero_spec_path)
            except FileNotFoundError:
                pass
        return flag

    def update_learnware(
        self, id: str, zip_path: str = None, semantic_spec: dict = None, check_status: int = None
    ) -> bool:
        """Update learnware with zip_path, semantic_specification and check_status.
           If the learnware supports heterogeneous market training, its HeteroMapTableSpecification is also updated.

        Parameters
        ----------
        id : str
            Learnware id
        zip_path : str, optional
            Filepath for learnware model, a zipped file.
        semantic_spec : dict, optional
            semantic_spec for new learnware, in dictionary format.
        check_status : int, optional
            A flag indicating whether the learnware is usable.

        Returns
        -------
        int
            The final learnware check_status.
        """
        final_status = super(HeteroMapTableOrganizer, self).update_learnware(id, zip_path, semantic_spec, check_status)
        if final_status == BaseChecker.USABLE_LEARWARE and len(self._get_hetero_learnware_ids(id)):
            self._update_learware_hetero_sepc(id)
        return final_status

    def _reload_learnware_hetero_spec(self, learnware_id):
        try:
            hetero_spec_path = os.path.join(self.hetero_specs_path, f"{learnware_id}.json")
            if os.path.exists(hetero_spec_path):
                hetero_spec = HeteroMapTableSpecification()
                hetero_spec.load(hetero_spec_path)
                self.learnware_list[learnware_id].update_stat_spec(hetero_spec.type, hetero_spec)
            else:
                self._update_learware_hetero_sepc(learnware_id)
            logger.info(f"Reload HeteroMapTableSpecification for hetero spec {learnware_id} succeed!")
        except Exception as err:
            logger.error(f"Reload HeteroMapTableSpecification for hetero spec {learnware_id} failed! due to {err}.")

    def reload_learnware(self, learnware_id: str):
        """Reload learnware into heterogeneous learnware market.
           If a corresponding HeteroMapTableSpecification exists, it is also reloaded.

        Parameters
        ----------
        learnware_id : str
            Learnware to be reloaded
        """
        super(HeteroMapTableOrganizer, self).reload_learnware(learnware_id)
        if len(self._get_hetero_learnware_ids(learnware_id)):
            self._reload_learnware_hetero_spec(learnware_id)

    def _update_learware_hetero_sepc(self, ids: Union[str, List[str]]):
        """Update learnware by ids, attempting to generate HeteroMapTableSpecification for them.

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of target learware
            List[str]: A list of ids of target learnwares
        """
        if isinstance(ids, str):
            ids = [ids]

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
                traceback.print_exc()
                logger.warning(f"Learnware {idx} generate HeteroMapTableSpecification failed!")

    def _get_hetero_learnware_ids(self, ids: Union[str, List[str]]) -> List[str]:
        """Get learnware ids that supports heterogeneous market training and search.

        Parameters
        ----------
        ids : Union[str, List[str]]
            Give a id or a list of ids
            str: id of target learware
            List[str]: A list of ids of target learnwares

        Returns
        -------
        List[str]
            Learnware ids
        """
        if isinstance(ids, str):
            ids = [ids]

        ret = []
        for idx in ids:
            spec = self.learnware_list[idx].get_specification()
            if is_hetero(stat_specs=spec.get_stat_spec(), semantic_spec=spec.get_semantic_spec()):
                ret.append(idx)
        return ret

    def generate_hetero_map_spec(self, user_info: BaseUserInfo) -> HeteroMapTableSpecification:
        """Generate HeteroMapTableSpecificaion based on user's input description and statistical information.

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info

        Returns
        -------
        HeteroMapTableSpecification
            The generated HeteroMapTableSpecification for user
        """
        user_stat_spec = user_info.stat_info["RKMETableSpecification"]
        user_features = user_info.get_semantic_spec()["Input"]["Description"]
        user_hetero_spec = self.market_mapping.hetero_mapping(user_stat_spec, user_features)
        return user_hetero_spec

    @staticmethod
    def train(learnware_list: List[Learnware], save_dir: str, **kwargs) -> HeteroMap:
        """Build the market mapping model using learnwares that supports heterogeneous market training.

        Parameters
        ----------
        learnware_list : List[Learnware]
            The learnware list to train the market mapping
        save_dir : str
            Filepath where the trained market mapping will be saved

        Returns
        -------
        HeteroMap
            The trained market mapping model
        """
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
