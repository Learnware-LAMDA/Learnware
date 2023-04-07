import os
from shutil import copyfile, rmtree
import zipfile
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Any, List, Union, Dict

from .base import BaseMarket, BaseUserInfo
from .database_ops import load_market_from_db, add_learnware_to_db, delete_learnware_from_db

from ..learnware import Learnware, get_learnware_from_config
from ..specification import RKMEStatSpecification, Specification
from ..logger import get_module_logger
from ..config import C

logger = get_module_logger("market", "INFO")


class EasyMarket(BaseMarket):
    def __init__(self):
        """Initializing an empty market"""
        self.learnware_list = {}  # id: Learnware
        self.learnware_zip_list = {}
        self.count = 0
        self.semantic_spec_list = C.semantic_specs
        self.reload_market()
        logger.info("Market Initialized!")

    def reload_market(self) -> bool:
        self.learnware_list, self.count = load_market_from_db()

    def check_learnware(self, learnware: Learnware) -> bool:
        """Check the utility of a learnware

        Parameters
        ----------
        learnware : Learnware

        Returns
        -------
        bool
            A flag indicating whether the learnware can be accepted.
        """
        try:
            spec_data = learnware.specification.stat_spec["RKME"].get_z()
            pred_spec = learnware.predict(spec_data)
        except Exception:
            logger.warning(f"The learnware [{learnware.id}-{learnware.name}] is not avaliable!")
            return False
        return True

    def add_learnware(self, learnware_name: str, zip_path: str, semantic_spec: dict) -> Tuple[str, bool]:
        """Add a learnware into the market.

        .. note::

            Given a prediction of a certain time, all signals before this time will be prepared well.


        Parameters
        ----------
        learnware_name : str
            Name of new learnware.
        model_path : str
            Filepath for learnware model, a zipped file.
        stat_spec_path : str
            Filepath for statistical specification, a '.npy' file.
            How to pass parameters requires further discussion.
        semantic_spec : dict
            semantic_spec for new learnware, in dictionary format.
        desc : str
            Brief desciption for new learnware.

        Returns
        -------
        Tuple[str, bool]
            str indicating model_id, bool indicating whether the learnware is added successfully.

        Raises
        ------
        FileNotFoundError
            file for model or statistical specification not found

        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError("Model or Stat_spec NOT Found.")

        """
        rkme_stat_spec = RKMEStatSpecification()
        rkme_stat_spec.load(stat_spec_path)
        stat_spec = {"RKME": rkme_stat_spec}
        specification = Specification(semantic_spec=semantic_spec, stat_spec=stat_spec)
        """

        id = "%08d" % (self.count)
        target_zip_dir = os.path.join(C.learnware_zip_pool_path, "%s.zip" % (id))
        target_folder_dir = os.path.join(C.learnware_folder_pool_path, id)
        copyfile(zip_path, target_zip_dir)
        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        config_file_dir = os.path.join(target_folder_dir, "learnware.yaml")

        new_learnware = get_learnware_from_config(id=id, semantic_spec=semantic_spec, file_config=config_file_dir)
        if new_learnware is None:
            os.rmdir(target_zip_dir)
            rmtree(target_folder_dir)
            return None, None
        else:
            self.learnware_list[id] = new_learnware
            self.learnware_zip_list[id] = target_zip_dir
            self.count += 1
            add_learnware_to_db(
                id,
                name=learnware_name,
                semantic_spec=semantic_spec,
                zip_path=target_folder_dir,
                folder_path=target_folder_dir,
            )
            return id, True

    def _calculate_rkme_spec_mixture_weight(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMEStatSpecification,
        intermediate_K: np.ndarray = None,
        intermediate_C: np.ndarray = None,
    ) -> Tuple[List[float], float]:
        """Calculate mixture weight for the learnware_list based on a user's rkme

        Parameters
        ----------
        learnware_list : List[Learnware]
            A list of existing learnwares
        user_rkme : RKMEStatSpecification
            User RKME statistical specification
        intermediate_K : np.ndarray, optional
            Intermediate kernel matrix K, by default None
        intermediate_C : np.ndarray, optional
            Intermediate inner product vector C, by default None

        Returns
        -------
        Tuple[List[float], float]
            The first is the list of mixture weights
            The second is the mmd dist between the mixture of learnware rkmes and the user's rkme
        """
        learnware_num = len(learnware_list)
        RKME_list = [learnware.specification.get_stat_spec_by_name("RKME") for learnware in learnware_list]

        if type(intermediate_K) == np.ndarray:
            K = intermediate_K
        else:
            K = np.zeros((learnware_num, learnware_num))
            for i in range(K.shape[0]):
                for j in range(K.shape[1]):
                    K[i, j] = RKME_list[i].inner_prod(RKME_list[j])

        if type(intermediate_C) == np.ndarray:
            C = intermediate_C
        else:
            C = np.zeros((learnware_num, 1))
            for i in range(C.shape[0]):
                C[i, 0] = user_rkme.inner_prod(RKME_list[i])

        K = torch.from_numpy(K).double().to(user_rkme.device)
        C = torch.from_numpy(C).double().to(user_rkme.device)

        # if nonnegative_beta:
        #    w = solve_qp(K, C).double().to(Phi_t.device)
        # else:
        weight = torch.linalg.inv(K + torch.eye(K.shape[0]).to(user_rkme.device) * 1e-5) @ C

        term1 = user_rkme.inner_prod(user_rkme)
        term2 = weight.T @ C
        term3 = weight.T @ K @ weight
        score = float(term1 - 2 * term2 + term3)

        return weight.detach().cpu().numpy().reshape(-1), score

    def _calculate_intermediate_K_and_C(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMEStatSpecification,
        intermediate_K: np.ndarray = None,
        intermediate_C: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Incrementally update the values of intermediate_K and intermediate_C

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares up till now
        user_rkme : RKMEStatSpecification
            User RKME statistical specification
        intermediate_K : np.ndarray, optional
            Intermediate kernel matrix K, by default None
        intermediate_C : np.ndarray, optional
            Intermediate inner product vector C, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The first is the intermediate value of K
            The second is the intermediate value of C
        """
        num = intermediate_K.shape[0] - 1
        RKME_list = [learnware.specification.get_stat_spec_by_name("RKME") for learnware in learnware_list]
        for i in range(intermediate_K.shape[0]):
            intermediate_K[num, i] = RKME_list[-1].inner_prod(RKME_list[i])
        intermediate_C[num, 0] = user_rkme.inner_prod(RKME_list[-1])
        return intermediate_K, intermediate_C

    def _search_by_rkme_spec_mixture(
        self, learnware_list: List[Learnware], user_rkme: RKMEStatSpecification, search_num: int
    ) -> Tuple[List[float], List[Learnware]]:
        """Get search_num learnwares with their mixture weight from the given learnware_list

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMEStatSpecification
            User RKME statistical specification
        search_num : int
            The number of the returned learnwares

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            The first is the list of weight
            The second is the list of Learnware
            The size of both list equals search_num
        """
        learnware_num = len(learnware_list)
        _, sorted_learnware_list = self._search_by_rkme_spec_single(learnware_list, user_rkme)
        flag_list = [0 for i in range(learnware_num)]
        mixture_list = []
        intermediate_K, intermediate_C = np.zeros((1, 1)), np.zeros((1, 1))

        for k in range(search_num):
            idx_min, score_min = -1, -1
            weight_min = None
            mixture_list.append(None)

            if k != 0:
                intermediate_K = np.c_[intermediate_K, np.zeros((k, 1))]
                intermediate_K = np.r_[intermediate_K, np.zeros((1, k + 1))]
                intermediate_C = np.r_[intermediate_C, np.zeros((1, 1))]

            for idx in range(len(sorted_learnware_list)):
                if flag_list[idx] == 0:
                    mixture_list[-1] = sorted_learnware_list[idx]
                    intermediate_K, intermediate_C = self._calculate_intermediate_K_and_C(
                        mixture_list, user_rkme, intermediate_K, intermediate_C
                    )
                    weight, score = self._calculate_rkme_spec_mixture_weight(
                        mixture_list, user_rkme, intermediate_K, intermediate_C
                    )
                    if idx_min == -1 or score < score_min:
                        idx_min, score_min, weight_min = idx, score, weight

            flag_list[idx_min] = 1
            mixture_list[-1] = sorted_learnware_list[idx_min]
            intermediate_K, intermediate_C = self._calculate_intermediate_K_and_C(
                mixture_list, user_rkme, intermediate_K, intermediate_C
            )

        return weight_min, mixture_list

    def _search_by_rkme_spec_single(
        self, learnware_list: List[Learnware], user_rkme: RKMEStatSpecification
    ) -> Tuple[List[float], List[Learnware]]:
        """Calculate the distances between learnwares in the given learnware_list and user_rkme

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMEStatSpecification
            user RKME statistical specification

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            the first is the list of mmd dist
            the second is the list of Learnware
            both lists are sorted by mmd dist
        """
        RKME_list = [learnware.specification.get_stat_spec_by_name("RKME") for learnware in learnware_list]
        mmd_dist_list = []
        for RKME in RKME_list:
            mmd_dist = RKME.dist(user_rkme)
            mmd_dist_list.append(mmd_dist)

        sorted_idx_list = sorted(range(len(learnware_list)), key=lambda k: mmd_dist_list[k])
        sorted_dist_list = [mmd_dist_list[idx] for idx in sorted_idx_list]
        sorted_learnware_list = [learnware_list[idx] for idx in sorted_idx_list]

        return sorted_dist_list, sorted_learnware_list

    def _search_by_semantic_description(
        self, learnware_list: List[Learnware], user_info: BaseUserInfo
    ) -> List[Learnware]:
        user_semantic_spec = user_info.get_semantic_spec()
        user_input_description = user_semantic_spec["Description"]["Values"]
        if not user_input_description:
            return []
        match_learnwares = []
        for learnware in learnware_list:
            learnware_name = learnware.get_name()
            if user_input_description in learnware_name:
                match_learnwares.append(learnware)
        return match_learnwares

    def _search_by_semantic_tags(self, learnware_list: List[Learnware], user_info: BaseUserInfo) -> List[Learnware]:
        def match_semantic_tags(semantic_spec1, semantic_spec2):
            if semantic_spec1.keys() != semantic_spec2.keys():
                raise Exception("semantic_spec key error".format(semantic_spec1.keys(), semantic_spec2.keys()))
            for key in semantic_spec1.keys():
                if semantic_spec1[key]["Type"] == "Class":
                    if semantic_spec1[key]["Values"] != semantic_spec2[key]["Values"]:
                        return False
                elif semantic_spec1[key]["Type"] == "Tag":
                    if not (set(semantic_spec1[key]["Values"]) & set(semantic_spec2[key]["Values"])):
                        return False
            return True

        match_learnwares = []
        for learnware in learnware_list:
            learnware_semantic_spec = learnware.get_specification().get_semantic_spec()
            user_semantic_spec = user_info.get_semantic_spec()
            if match_semantic_tags(learnware_semantic_spec, user_semantic_spec):
                match_learnwares.append(learnware)
        return match_learnwares

    def search_learnware(
        self, user_info: BaseUserInfo, search_num=3
    ) -> Tuple[List[float], List[Learnware], List[Learnware]]:
        """Search learnwares based on user_info

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        search_num : int
            The number of the returned learnwares

        Returns
        -------
        Tuple[List[float], List[Learnware], List[float], List[Learnware]]
            the first is the sorted list of rkme dist
            the second is the sorted list of Learnware (single) by the rkme dist
            the third is the list of Learnware (mixture), the size is search_num
        """
        learnware_list = [self.learnware_list[key] for key in self.learnware_list]
        learnware_list_tags = self._search_by_semantic_tags(learnware_list, user_info)
        learnware_list_description = self._search_by_semantic_description(learnware_list, user_info)
        learnware_list = list(set(learnware_list_tags + learnware_list_description))

        if "RKME" not in user_info.stat_info:
            return None, learnware_list, None
        else:
            user_rkme = user_info.stat_info["RKME"]
            sorted_dist_list, single_learnware_list = self._search_by_rkme_spec_single(learnware_list, user_rkme)
            weight_list, mixture_learnware_list = self._search_by_rkme_spec_mixture(
                learnware_list, user_rkme, search_num
            )
            return sorted_dist_list, single_learnware_list, mixture_learnware_list

    def delete_learnware(self, id: str) -> bool:
        if not id in self.learnware_list:
            raise Exception("Learnware id:'{}' NOT Found!".format(id))

        self.learnware_list.pop(id)
        self.learnware_zip_list.pop(id)
        delete_learnware_from_db(id)
        return True

    def get_semantic_spec_list(self) -> dict:
        return self.semantic_spec_list

    def get_learnware_by_ids(self, id: str):
        pass

    def get_learnware_path_by_ids(self, id: str) -> str:
        pass

    def __len__(self):
        return len(self.learnware_list.keys())

    def _get_ids(self, top=None):
        if top is None:
            return list(self.learnware_list.keys())
        else:
            return list(self.learnware_list.keys())[:top]
