import os
from shutil import copyfile, rmtree
import zipfile
import torch
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from typing import Tuple, Any, List, Union, Dict

from .base import BaseMarket, BaseUserInfo
from .database_ops import load_market_from_db, add_learnware_to_db, delete_learnware_from_db, clear_learnware_table

from ..learnware import Learnware, get_learnware_from_dirpath
from ..specification import RKMEStatSpecification, Specification
from ..logger import get_module_logger
from ..config import C

logger = get_module_logger("market", "INFO")


class EasyMarket(BaseMarket):
    def __init__(self, rebuild: bool = False):
        """Initialize Learnware Market.
        Automatically reload from db if available.
        Build an empty db otherwise.

        Parameters
        ----------
        rebuild : bool, optional
            Clear current database if set to True, by default False
            !!! Do NOT set to True unless highly necessary !!!
        """
        self.learnware_list = {}  # id: Learnware
        self.learnware_zip_list = {}
        self.learnware_folder_list = {}
        self.count = 0
        self.semantic_spec_list = C.semantic_specs
        self.reload_market(rebuild=rebuild)  # Automatically reload the market
        logger.info("Market Initialized!")

    def reload_market(self, rebuild: bool = False) -> bool:
        if rebuild:
            logger.warning("Warning! You are trying to clear current database!")
            clear_learnware_table()
            rmtree(C.learnware_pool_path)

        os.makedirs(C.learnware_pool_path, exist_ok=True)
        os.makedirs(C.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(C.learnware_folder_pool_path, exist_ok=True)
        self.learnware_list, self.learnware_zip_list, self.learnware_folder_list, self.count = load_market_from_db()

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
            spec_data = learnware.specification.stat_spec["RKMEStatSpecification"].get_z()
            pred_spec = learnware.predict(spec_data)
        except Exception:
            logger.warning(f"The learnware [{learnware.id}-{learnware.name}] is not avaliable!")
            return False
        return True

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
        Tuple[str, bool]
            - str indicating model_id
            - bool indicating whether the learnware is added successfully.

        """
        if not os.path.exists(zip_path):
            logger.warning("Zip Path NOT Found! Fail to add learnware.")
            return None, False

        try:
            if len(semantic_spec["Data"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Data.")
                return None, False
            if len(semantic_spec["Task"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Task.")
                return None, False
            if len(semantic_spec["Device"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please choose Device.")
                return None, False
            if len(semantic_spec["Name"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please provide Name.")
                return None, False
            if len(semantic_spec["Description"]["Values"]) == 0 and len(semantic_spec["Scenario"]["Values"]) == 0:
                logger.warning("Illegal semantic specification, please provide Scenario or Description.")
                return None, False
        except:
            logger.warning("Illegal semantic specification, some keys are missing.")
            return None, False

        logger.info("Get new learnware from %s" % (zip_path))
        id = "%08d" % (self.count)
        target_zip_dir = os.path.join(C.learnware_zip_pool_path, "%s.zip" % (id))
        target_folder_dir = os.path.join(C.learnware_folder_pool_path, id)
        copyfile(zip_path, target_zip_dir)

        with zipfile.ZipFile(target_zip_dir, "r") as z_file:
            z_file.extractall(target_folder_dir)
        logger.info("Learnware move to %s, and unzip to %s" % (target_zip_dir, target_folder_dir))
        try:
            new_learnware = get_learnware_from_dirpath(
                id=id, semantic_spec=semantic_spec, learnware_dirpath=target_folder_dir
            )
        except:
            new_learnware = None

        if new_learnware is None:
            try:
                os.remove(target_zip_dir)
                rmtree(target_folder_dir)
            except:
                pass
            return None, False
        else:
            self.learnware_list[id] = new_learnware
            self.learnware_zip_list[id] = target_zip_dir
            self.learnware_folder_list[id] = target_folder_dir
            self.count += 1
            add_learnware_to_db(
                id,
                semantic_spec=semantic_spec,
                zip_path=target_zip_dir,
                folder_path=target_folder_dir,
            )
            return id, True

    def _convert_dist_to_score(self, dist_list: List[float]) -> List[float]:
        """Convert mmd dist list into min_max score list

        Parameters
        ----------
        dist_list : List[float]
            The list of mmd distances from learnware rkmes to user rkme

        Returns
        -------
        List[float]
            The list of min_max scores of each learnware
        """
        if max(dist_list) == min(dist_list):
            return [1 for dist in dist_list]
        else:
            return [(max(dist_list) - dist) / (max(dist_list) - min(dist_list)) for dist in dist_list]

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
        RKME_list = [
            learnware.specification.get_stat_spec_by_name("RKMEStatSpecification") for learnware in learnware_list
        ]

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

        # beta can be negative
        # weight = torch.linalg.inv(K + torch.eye(K.shape[0]).to(user_rkme.device) * 1e-5) @ C

        # beta must be nonnegative
        n = K.shape[0]
        P = matrix(K.cpu().numpy())
        q = matrix(-C.cpu().numpy())
        G = matrix(-np.eye(n))
        h = matrix(np.zeros((n, 1)))
        A = matrix(np.ones((1, n)))
        b = matrix(np.ones((1, 1)))
        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h, A, b)
        weight = np.array(sol["x"])
        weight = torch.from_numpy(weight).reshape(-1).double().to(user_rkme.device)

        term1 = user_rkme.inner_prod(user_rkme)
        # print('weight:', weight.shape, 'C:', C.shape)
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
        RKME_list = [
            learnware.specification.get_stat_spec_by_name("RKMEStatSpecification") for learnware in learnware_list
        ]
        for i in range(intermediate_K.shape[0]):
            intermediate_K[num, i] = RKME_list[-1].inner_prod(RKME_list[i])
        intermediate_C[num, 0] = user_rkme.inner_prod(RKME_list[-1])
        return intermediate_K, intermediate_C
    
    def _search_by_rkme_spec_mixture_auto(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMEStatSpecification,
        max_search_num: int,
        weight_cutoff: float = 0.9
    ) -> Tuple[List[float], List[Learnware]]:
        """Select learnwares based on a total mixture ratio, then recalculate their mixture weights

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMEStatSpecification
            User RKME statistical specification
        max_search_num : int
            The maximum number of the returned learnwares
        weight_cutoff : float, optional
            The ratio for selecting out the mose relevant learnwares, by default 0.9

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            The first is the list of weight
            The second is the list of Learnware
        """
        learnware_num = len(learnware_list)
        if learnware_num == 0:
            return [], []
        if learnware_num < max_search_num:
            logger.warning("Available Learnware num less than search_num!")
            max_search_num = learnware_num

        weight, _ = self._calculate_rkme_spec_mixture_weight(learnware_list, user_rkme)
        sort_by_weight_idx_list = sorted(range(learnware_num), key=lambda k: weight[k])
        
        weight_sum = 0
        mixture_list = []
        for idx in sort_by_weight_idx_list:
            weight_sum += sort_by_weight_idx_list[idx]
            if weight_sum <= weight_cutoff:
                mixture_list.append(learnware_list[idx])

        if len(mixture_list) > max_search_num:
            mixture_list = mixture_list[:max_search_num]
        
        mixture_weight, _ = self._calculate_rkme_spec_mixture_weight(mixture_list, user_rkme)
        return mixture_weight, mixture_list

    def _filter_by_rkme_spec_single(self, sorted_score_list: List[float], learnware_list: List[Learnware], filter_score=60, min_num=15) -> Tuple[List[float], List[Learnware]]:
        """Filter search result of _search_by_rkme_spec_single

        Parameters
        ----------
        sorted_score_list : List[float]
            The list of score transformed by mmd dist
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            the first is the list of score
            the second is the list of Learnware
        """
        idx = min(min_num, len(learnware_list))
        while idx < len(learnware_list):
            if sorted_score_list[idx] < filter_score:
                break
            idx = idx + 1
        return sorted_score_list[:idx], learnware_list[:idx]
    
    def _search_by_rkme_spec_mixture(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMEStatSpecification,
        max_search_num: int,
        score_cutoff: float = 0.01,
    ) -> Tuple[List[float], List[Learnware]]:
        """Greedily match learnwares such that their mixture become more and more closer to user's rkme  

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMEStatSpecification
            User RKME statistical specification
        max_search_num : int
            The maximum number of the returned learnwares
        score_cutof: float
            The minimum mmd dist as threshold to stop further rkme_spec matching

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            The first is the list of weight
            The second is the list of Learnware
        """
        learnware_num = len(learnware_list)
        if learnware_num == 0:
            return [], []
        if learnware_num < max_search_num:
            logger.warning("Available Learnware num less than search_num!")
            max_search_num = learnware_num

        flag_list = [0 for _ in range(learnware_num)]
        mixture_list = []
        intermediate_K, intermediate_C = np.zeros((1, 1)), np.zeros((1, 1))

        for k in range(max_search_num):
            idx_min, score_min = -1, -1
            weight_min = None
            mixture_list.append(None)

            if k != 0:
                intermediate_K = np.c_[intermediate_K, np.zeros((k, 1))]
                intermediate_K = np.r_[intermediate_K, np.zeros((1, k + 1))]
                intermediate_C = np.r_[intermediate_C, np.zeros((1, 1))]

            for idx in range(len(learnware_list)):
                if flag_list[idx] == 0:
                    mixture_list[-1] = learnware_list[idx]
                    intermediate_K, intermediate_C = self._calculate_intermediate_K_and_C(
                        mixture_list, user_rkme, intermediate_K, intermediate_C
                    )
                    weight, score = self._calculate_rkme_spec_mixture_weight(
                        mixture_list, user_rkme, intermediate_K, intermediate_C
                    )
                    if idx_min == -1 or score < score_min:
                        idx_min, score_min, weight_min = idx, score, weight

            mixture_list[-1] = learnware_list[idx_min]
            if score_min < score_cutoff:
                break
            else:
                flag_list[idx_min] = 1
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
        RKME_list = [
            learnware.specification.get_stat_spec_by_name("RKMEStatSpecification") for learnware in learnware_list
        ]
        mmd_dist_list = []
        for RKME in RKME_list:
            mmd_dist = RKME.dist(user_rkme)
            mmd_dist_list.append(mmd_dist)

        sorted_idx_list = sorted(range(len(learnware_list)), key=lambda k: mmd_dist_list[k])
        sorted_dist_list = [mmd_dist_list[idx] for idx in sorted_idx_list]
        sorted_learnware_list = [learnware_list[idx] for idx in sorted_idx_list]

        return sorted_dist_list, sorted_learnware_list

    def _search_by_semantic_spec(self, learnware_list: List[Learnware], user_info: BaseUserInfo) -> List[Learnware]:
        def match_semantic_spec(semantic_spec1, semantic_spec2):
            if semantic_spec1.keys() != semantic_spec2.keys():
                # raise Exception("semantic_spec key error")
                logger.warning("semantic_spec key error!")
                return False
            for key in semantic_spec1.keys():
                if len(semantic_spec1[key]["Values"]) == 0:
                    continue
                if len(semantic_spec2[key]["Values"]) == 0:
                    continue
                v1 = semantic_spec1[key]["Values"]
                v2 = semantic_spec2[key]["Values"]
                if semantic_spec1[key]["Type"] == "Class":
                    if isinstance(v1, list):
                        v1 = v1[0]
                    if isinstance(v2, list):
                        v2 = v2[0]
                    if v1 != v2:
                        return False
                elif semantic_spec1[key]["Type"] == "Tag":
                    if not (set(v1) & set(v2)):
                        return False
                elif semantic_spec1[key]["Type"] == "Name":
                    if v2 not in v1 and v2 not in semantic_spec1["Description"]["Values"]:
                        return False
            return True

        match_learnwares = []
        for learnware in learnware_list:
            learnware_semantic_spec = learnware.get_specification().get_semantic_spec()
            user_semantic_spec = user_info.get_semantic_spec()
            if match_semantic_spec(learnware_semantic_spec, user_semantic_spec):
                match_learnwares.append(learnware)
        return match_learnwares

    def search_learnware(
        self, user_info: BaseUserInfo, max_search_num=5
    ) -> Tuple[List[float], List[Learnware], List[Learnware]]:
        """Search learnwares based on user_info

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        max_search_num : int
            The maximum number of the returned learnwares

        Returns
        -------
        Tuple[List[float], List[Learnware], List[float], List[Learnware]]
            the first is the sorted list of rkme dist
            the second is the sorted list of Learnware (single) by the rkme dist
            the third is the list of Learnware (mixture), the size is search_num
        """
        learnware_list = [self.learnware_list[key] for key in self.learnware_list]
        learnware_list = self._search_by_semantic_spec(learnware_list, user_info)
        # learnware_list = list(set(learnware_list_tags + learnware_list_description))

        if "RKMEStatSpecification" not in user_info.stat_info:
            return None, learnware_list, None
        elif len(learnware_list) == 0:
            return [], [], []
        else:
            user_rkme = user_info.stat_info["RKMEStatSpecification"]
            sorted_dist_list, single_learnware_list = self._search_by_rkme_spec_single(learnware_list, user_rkme)
            sorted_score_list = self._convert_dist_to_score(sorted_dist_list)
            sorted_score_list, single_learnware_list = self._filter_by_rkme_spec_single(sorted_score_list, single_learnware_list)
            weight_list, mixture_learnware_list = self._search_by_rkme_spec_mixture(
                learnware_list, user_rkme, max_search_num
            )
            return sorted_score_list, single_learnware_list, mixture_learnware_list

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
        delete_learnware_from_db(id)

        return True

    def get_semantic_spec_list(self) -> dict:
        return self.semantic_spec_list

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

    def __len__(self):
        return len(self.learnware_list.keys())

    def _get_ids(self, top=None):
        if top is None:
            return list(self.learnware_list.keys())
        else:
            return list(self.learnware_list.keys())[:top]
