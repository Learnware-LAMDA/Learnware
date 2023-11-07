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

from .base import LearnwareMarket, BaseUserInfo
from .database_ops import DatabaseOperations

from .. import utils
from ..config import C as conf
from ..logger import get_module_logger
from ..learnware import Learnware, get_learnware_from_dirpath
from ..specification import RKMETableSpecification, Specification


logger = get_module_logger("market", "INFO")


class EasyMarket(LearnwareMarket):
    """EasyMarket provide an easy and simple implementation for LearnwareMarket
    - EasyMarket stores learnwares with file system and database
    - EasyMarket search the learnwares with the match of semantical tag and the statistical RKME
    - EasyMarket does not support the search between heterogeneous features learnwars
    """

    INVALID_LEARNWARE = -1
    NONUSABLE_LEARNWARE = 0
    USABLE_LEARWARE = 1

    def __init__(self, market_id: str = "default", rebuild: bool = False):
        """Initialize Learnware Market.
        Automatically reload from db if available.
        Build an empty db otherwise.

        Parameters
        ----------
        market_id : str, optional, by default 'default'
            The unique market id for market database

        rebuild : bool, optional
            Clear current database if set to True, by default False
            !!! Do NOT set to True unless highly necessary !!!
        """
        self.market_id = market_id
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
        self.reload_market(rebuild=rebuild)  # Automatically reload the market
        logger.info("Market Initialized!")

    def reload_market(self, rebuild: bool = False) -> bool:
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

    @classmethod
    def check_learnware(cls, learnware: Learnware) -> int:
        """Check the utility of a learnware

        Parameters
        ----------
        learnware : Learnware

        Returns
        -------
        int
            A flag indicating whether the learnware can be accepted.
            - The INVALID_LEARNWARE denotes the learnware does not pass the check
            - The NOPREDICTION_LEARNWARE denotes the learnware pass the check but cannot make prediction due to some env dependency
            - The NOPREDICTION_LEARNWARE denotes the leanrware pass the check and can make prediction
        """

        semantic_spec = learnware.get_specification().get_semantic_spec()

        try:
            # check model instantiation
            learnware.instantiate_model()

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"The learnware [{learnware.id}] is instantiated failed! Due to {e}")
            return cls.NONUSABLE_LEARNWARE

        try:
            learnware_model = learnware.get_model()

            # check input shape
            if semantic_spec["Data"]["Values"][0] == "Table":
                input_shape = (semantic_spec["Input"]["Dimension"],)
            else:
                input_shape = learnware_model.input_shape
                pass

            # check rkme dimension
            stat_spec = learnware.get_specification().get_stat_spec_by_name("RKMETableSpecification")
            if stat_spec is not None:
                if stat_spec.get_z().shape[1:] != input_shape:
                    logger.warning(f"The learnware [{learnware.id}] input dimension mismatch with stat specification")
                    return cls.NONUSABLE_LEARNWARE
                pass

            inputs = np.random.randn(10, *input_shape)
            outputs = learnware.predict(inputs)

            # check output
            if outputs.ndim == 1:
                outputs = outputs.reshape(-1, 1)
                pass

            if semantic_spec["Task"]["Values"][0] in ("Classification", "Regression", "Feature Extraction"):
                # check output type
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.detach().cpu().numpy()
                if not isinstance(outputs, np.ndarray):
                    logger.warning(f"The learnware [{learnware.id}] output must be np.ndarray or torch.Tensor")
                    return cls.NONUSABLE_LEARNWARE

                # check output shape
                output_dim = int(semantic_spec["Output"]["Dimension"])
                if outputs[0].shape[0] != output_dim:
                    logger.warning(f"The learnware [{learnware.id}] input and output dimention is error")
                    return cls.NONUSABLE_LEARNWARE
                pass
            else:
                if outputs.shape[1:] != learnware_model.output_shape:
                    logger.warning(f"The learnware [{learnware.id}] input and output dimention is error")
                    return cls.NONUSABLE_LEARNWARE

        except Exception as e:
            logger.exception
            logger.warning(f"The learnware [{learnware.id}] prediction is not avaliable! Due to {repr(e)}")
            raise e
            return cls.NONUSABLE_LEARNWARE

        return cls.USABLE_LEARWARE

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

    def _convert_dist_to_score(
        self, dist_list: List[float], dist_epsilon: float = 0.01, min_score: float = 0.92
    ) -> List[float]:
        """Convert mmd dist list into min_max score list

        Parameters
        ----------
        dist_list : List[float]
            The list of mmd distances from learnware rkmes to user rkme
        dist_epsilon: float
            The paramter for converting mmd dist to score
        min_score: float
            The minimum score for maximum returned score

        Returns
        -------
        List[float]
            The list of min_max scores of each learnware
        """
        if len(dist_list) == 0:
            return []

        min_dist, max_dist = min(dist_list), max(dist_list)
        if min_dist == max_dist:
            return [1 for dist in dist_list]
        else:
            max_score = (max_dist - min_dist) / (max_dist - dist_epsilon)

            if min_dist < dist_epsilon:
                dist_epsilon = min_dist
            elif max_score < min_score:
                dist_epsilon = max_dist - (max_dist - min_dist) / min_score

            return [(max_dist - dist) / (max_dist - dist_epsilon) for dist in dist_list]

    def _calculate_rkme_spec_mixture_weight(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMETableSpecification,
        intermediate_K: np.ndarray = None,
        intermediate_C: np.ndarray = None,
    ) -> Tuple[List[float], float]:
        """Calculate mixture weight for the learnware_list based on a user's rkme

        Parameters
        ----------
        learnware_list : List[Learnware]
            A list of existing learnwares
        user_rkme : RKMETableSpecification
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
            learnware.specification.get_stat_spec_by_name("RKMETableSpecification") for learnware in learnware_list
        ]

        if type(intermediate_K) == np.ndarray:
            K = intermediate_K
        else:
            K = np.zeros((learnware_num, learnware_num))
            for i in range(K.shape[0]):
                K[i, i] = RKME_list[i].inner_prod(RKME_list[i])
                for j in range(i + 1, K.shape[0]):
                    K[i, j] = K[j, i] = RKME_list[i].inner_prod(RKME_list[j])

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
        score = user_rkme.inner_prod(user_rkme) + 2 * sol["primal objective"]

        return weight.detach().cpu().numpy().reshape(-1), score

    def _calculate_intermediate_K_and_C(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMETableSpecification,
        intermediate_K: np.ndarray = None,
        intermediate_C: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Incrementally update the values of intermediate_K and intermediate_C

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares up till now
        user_rkme : RKMETableSpecification
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
            learnware.specification.get_stat_spec_by_name("RKMETableSpecification") for learnware in learnware_list
        ]
        for i in range(intermediate_K.shape[0]):
            intermediate_K[num, i] = RKME_list[-1].inner_prod(RKME_list[i])
        intermediate_C[num, 0] = user_rkme.inner_prod(RKME_list[-1])
        return intermediate_K, intermediate_C

    def _search_by_rkme_spec_mixture_auto(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMETableSpecification,
        max_search_num: int,
        weight_cutoff: float = 0.98,
    ) -> Tuple[float, List[float], List[Learnware]]:
        """Select learnwares based on a total mixture ratio, then recalculate their mixture weights

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMETableSpecification
            User RKME statistical specification
        max_search_num : int
            The maximum number of the returned learnwares
        weight_cutoff : float, optional
            The ratio for selecting out the mose relevant learnwares, by default 0.9

        Returns
        -------
        Tuple[float, List[float], List[Learnware]]
            The first is the mixture mmd dist
            The second is the list of weight
            The third is the list of Learnware
        """
        learnware_num = len(learnware_list)
        if learnware_num == 0:
            return [], []
        if learnware_num < max_search_num:
            logger.warning("Available Learnware num less than search_num!")
            max_search_num = learnware_num

        weight, _ = self._calculate_rkme_spec_mixture_weight(learnware_list, user_rkme)
        sort_by_weight_idx_list = sorted(range(learnware_num), key=lambda k: weight[k], reverse=True)

        weight_sum = 0
        mixture_list = []
        for idx in sort_by_weight_idx_list:
            weight_sum += weight[idx]
            if weight_sum <= weight_cutoff:
                mixture_list.append(learnware_list[idx])
            else:
                break

        if len(mixture_list) <= 1:
            mixture_list = [learnware_list[sort_by_weight_idx_list[0]]]
            mixture_weight = [1]
            mmd_dist = user_rkme.dist(mixture_list[0].specification.get_stat_spec_by_name("RKMETableSpecification"))
        else:
            if len(mixture_list) > max_search_num:
                mixture_list = mixture_list[:max_search_num]
            mixture_weight, mmd_dist = self._calculate_rkme_spec_mixture_weight(mixture_list, user_rkme)

        return mmd_dist, mixture_weight, mixture_list

    def _filter_by_rkme_spec_single(
        self,
        sorted_score_list: List[float],
        learnware_list: List[Learnware],
        filter_score: float = 0.5,
        min_num: int = 15,
    ) -> Tuple[List[float], List[Learnware]]:
        """Filter search result of _search_by_rkme_spec_single

        Parameters
        ----------
        sorted_score_list : List[float]
            The list of score transformed by mmd dist
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        filter_score: float
            The learnware whose score is lower than filter_score will be filtered
        min_num: int
            The minimum number of returned learnwares

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

    def _filter_by_rkme_spec_dimension(
        self, learnware_list: List[Learnware], user_rkme: RKMETableSpecification
    ) -> List[Learnware]:
        """Filter learnwares whose rkme dimension different from user_rkme

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMETableSpecification
            User RKME statistical specification

        Returns
        -------
        List[Learnware]
            Learnwares whose rkme dimensions equal user_rkme in user_info
        """
        filtered_learnware_list = []
        user_rkme_dim = str(list(user_rkme.get_z().shape)[1:])

        for learnware in learnware_list:
            rkme = learnware.specification.get_stat_spec_by_name("RKMETableSpecification")
            rkme_dim = str(list(rkme.get_z().shape)[1:])
            if rkme_dim == user_rkme_dim:
                filtered_learnware_list.append(learnware)

        return filtered_learnware_list

    def _search_by_rkme_spec_mixture_greedy(
        self,
        learnware_list: List[Learnware],
        user_rkme: RKMETableSpecification,
        max_search_num: int,
        score_cutoff: float = 0.001,
    ) -> Tuple[float, List[float], List[Learnware]]:
        """Greedily match learnwares such that their mixture become closer and closer to user's rkme

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMETableSpecification
            User RKME statistical specification
        max_search_num : int
            The maximum number of the returned learnwares
        score_cutof: float
            The minimum mmd dist as threshold to stop further rkme_spec matching

        Returns
        -------
        Tuple[float, List[float], List[Learnware]]
            The first is the mixture mmd dist
            The second is the list of weight
            The third is the list of Learnware
        """
        learnware_num = len(learnware_list)
        if learnware_num == 0:
            return None, [], []
        if learnware_num < max_search_num:
            logger.warning("Available Learnware num less than search_num!")
            max_search_num = learnware_num

        flag_list = [0 for _ in range(learnware_num)]
        mixture_list, mmd_dist = [], None
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

            mmd_dist = score_min
            mixture_list[-1] = learnware_list[idx_min]
            if score_min < score_cutoff:
                break
            else:
                flag_list[idx_min] = 1
                intermediate_K, intermediate_C = self._calculate_intermediate_K_and_C(
                    mixture_list, user_rkme, intermediate_K, intermediate_C
                )

        return mmd_dist, weight_min, mixture_list

    def _search_by_rkme_spec_single(
        self, learnware_list: List[Learnware], user_rkme: RKMETableSpecification
    ) -> Tuple[List[float], List[Learnware]]:
        """Calculate the distances between learnwares in the given learnware_list and user_rkme

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares whose mixture approximates the user's rkme
        user_rkme : RKMETableSpecification
            user RKME statistical specification

        Returns
        -------
        Tuple[List[float], List[Learnware]]
            the first is the list of mmd dist
            the second is the list of Learnware
            both lists are sorted by mmd dist
        """
        RKME_list = [
            learnware.specification.get_stat_spec_by_name("RKMETableSpecification") for learnware in learnware_list
        ]
        mmd_dist_list = []
        for RKME in RKME_list:
            mmd_dist = RKME.dist(user_rkme)
            mmd_dist_list.append(mmd_dist)

        sorted_idx_list = sorted(range(len(learnware_list)), key=lambda k: mmd_dist_list[k])
        sorted_dist_list = [mmd_dist_list[idx] for idx in sorted_idx_list]
        sorted_learnware_list = [learnware_list[idx] for idx in sorted_idx_list]

        return sorted_dist_list, sorted_learnware_list

    def _search_by_semantic_spec_exact(
        self, learnware_list: List[Learnware], user_info: BaseUserInfo
    ) -> List[Learnware]:
        def match_semantic_spec(semantic_spec1, semantic_spec2):
            """
            semantic_spec1: semantic spec input by user
            semantic_spec2: semantic spec in database
            """
            if semantic_spec1.keys() != semantic_spec2.keys():
                # sematic spec in database may contain more keys than user input
                pass

            name2 = semantic_spec2["Name"]["Values"].lower()
            description2 = semantic_spec2["Description"]["Values"].lower()

            for key in semantic_spec1.keys():
                v1 = semantic_spec1[key]["Values"]
                v2 = semantic_spec2[key]["Values"]

                if len(v1) == 0:
                    # user input is empty, no need to search
                    continue

                if key in ("Name", "Description"):
                    v1 = v1.lower()
                    if v1 not in name2 and v1 not in description2:
                        return False
                    pass
                else:
                    if len(v2) == 0:
                        # user input contains some key that is not in database
                        return False

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
                        pass
                    pass
                pass

            return True

        match_learnwares = []
        for learnware in learnware_list:
            learnware_semantic_spec = learnware.get_specification().get_semantic_spec()
            user_semantic_spec = user_info.get_semantic_spec()
            if match_semantic_spec(user_semantic_spec, learnware_semantic_spec):
                match_learnwares.append(learnware)
        logger.info("semantic_spec search: choose %d from %d learnwares" % (len(match_learnwares), len(learnware_list)))
        return match_learnwares

    def _search_by_semantic_spec_fuzz(
        self, learnware_list: List[Learnware], user_info: BaseUserInfo, max_num: int = 50000, min_score: float = 75.0
    ) -> List[Learnware]:
        """Search learnware by fuzzy matching of semantic spec

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list of learnwares
        user_info : BaseUserInfo
            user_info contains semantic_spec
        max_num : int, optional
            maximum number of learnwares returned, by default 50000
        min_score : float, optional
            Minimum fuzzy matching score of learnwares returned, by default 30.0

        Returns
        -------
        List[Learnware]
            The list of returned learnwares
        """

        def _match_semantic_spec_tag(semantic_spec1, semantic_spec2) -> bool:
            """Judge if tags of two semantic specs are consistent

            Parameters
            ----------
            semantic_spec1 :
                semantic spec input by user
            semantic_spec2 :
                semantic spec in database

            Returns
            -------
            bool
                consistent (True) or not consistent (False)
            """
            for key in semantic_spec1.keys():
                v1 = semantic_spec1[key]["Values"]
                v2 = semantic_spec2[key]["Values"]

                if len(v1) == 0:
                    # user input is empty, no need to search
                    continue

                if key not in "Name":
                    if len(v2) == 0:
                        # user input contains some key that is not in database
                        return False

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
            return True

        matched_learnware_tag = []
        final_result = []
        user_semantic_spec = user_info.get_semantic_spec()

        for learnware in learnware_list:
            learnware_semantic_spec = learnware.get_specification().get_semantic_spec()
            if _match_semantic_spec_tag(user_semantic_spec, learnware_semantic_spec):
                matched_learnware_tag.append(learnware)

        if len(matched_learnware_tag) > 0:
            if "Name" in user_semantic_spec:
                name_user = user_semantic_spec["Name"]["Values"].lower()
                if len(name_user) > 0:
                    # Exact search
                    name_list = [
                        learnware.get_specification().get_semantic_spec()["Name"]["Values"].lower()
                        for learnware in matched_learnware_tag
                    ]
                    des_list = [
                        learnware.get_specification().get_semantic_spec()["Description"]["Values"].lower()
                        for learnware in matched_learnware_tag
                    ]

                    matched_learnware_exact = []
                    for i in range(len(name_list)):
                        if name_user in name_list[i] or name_user in des_list[i]:
                            matched_learnware_exact.append(matched_learnware_tag[i])

                    if len(matched_learnware_exact) == 0:
                        # Fuzzy search
                        matched_learnware_fuzz, fuzz_scores = [], []
                        for i in range(len(name_list)):
                            score_name = fuzz.partial_ratio(name_user, name_list[i])
                            score_des = fuzz.partial_ratio(name_user, des_list[i])
                            final_score = max(score_name, score_des)
                            if final_score >= min_score:
                                matched_learnware_fuzz.append(matched_learnware_tag[i])
                                fuzz_scores.append(final_score)

                        # Sort by score
                        sort_idx = sorted(list(range(len(fuzz_scores))), key=lambda k: fuzz_scores[k], reverse=True)[
                            :max_num
                        ]
                        final_result = [matched_learnware_fuzz[idx] for idx in sort_idx]
                    else:
                        final_result = matched_learnware_exact
                else:
                    final_result = matched_learnware_tag
            else:
                final_result = matched_learnware_tag

        logger.info("semantic_spec search: choose %d from %d learnwares" % (len(final_result), len(learnware_list)))
        return final_result

    def search_learnware(
        self, user_info: BaseUserInfo, max_search_num: int = 5, search_method: str = "greedy"
    ) -> Tuple[List[float], List[Learnware], float, List[Learnware]]:
        """Search learnwares based on user_info

        Parameters
        ----------
        user_info : BaseUserInfo
            user_info contains semantic_spec and stat_info
        max_search_num : int
            The maximum number of the returned learnwares

        Returns
        -------
        Tuple[List[float], List[Learnware], float, List[Learnware]]
            the first is the sorted list of rkme dist
            the second is the sorted list of Learnware (single) by the rkme dist
            the third is the score of Learnware (mixture)
            the fourth is the list of Learnware (mixture), the size is search_num
        """
        learnware_list = [self.learnware_list[key] for key in self.learnware_list]
        # learnware_list = self._search_by_semantic_spec_exact(learnware_list, user_info)
        # if len(learnware_list) == 0:
        logger.info(f"stat_info in user_info: {user_info.stat_info}")
        learnware_list = self._search_by_semantic_spec_fuzz(learnware_list, user_info)
        logger.info(f"Number of learnwares after semantic fuzzy search: {len(learnware_list)}")

        if "RKMETableSpecification" not in user_info.stat_info:
            return None, learnware_list, 0.0, None
        elif len(learnware_list) == 0:
            return [], [], 0.0, []
        else:
            user_rkme = user_info.stat_info["RKMETableSpecification"]
            learnware_list = self._filter_by_rkme_spec_dimension(learnware_list, user_rkme)
            logger.info(f"After filter by rkme dimension, learnware_list length is {len(learnware_list)}")

            sorted_dist_list, single_learnware_list = self._search_by_rkme_spec_single(learnware_list, user_rkme)
            if search_method == "auto":
                mixture_dist, weight_list, mixture_learnware_list = self._search_by_rkme_spec_mixture_auto(
                    learnware_list, user_rkme, max_search_num
                )
            elif search_method == "greedy":
                mixture_dist, weight_list, mixture_learnware_list = self._search_by_rkme_spec_mixture_greedy(
                    learnware_list, user_rkme, max_search_num
                )
            else:
                logger.warning("f{search_method} not supported!")
                mixture_dist = None
                weight_list = []
                mixture_learnware_list = []

            if mixture_dist is None:
                sorted_score_list = self._convert_dist_to_score(sorted_dist_list)
                mixture_score = None
            else:
                merge_score_list = self._convert_dist_to_score(sorted_dist_list + [mixture_dist])
                sorted_score_list = merge_score_list[:-1]
                mixture_score = merge_score_list[-1]

            logger.info(f"After search by rkme spec, learnware_list length is {len(learnware_list)}")
            # filter learnware with low score
            sorted_score_list, single_learnware_list = self._filter_by_rkme_spec_single(
                sorted_score_list, single_learnware_list
            )

            logger.info(f"After filter by rkme spec, learnware_list length is {len(learnware_list)}")
            return sorted_score_list, single_learnware_list, mixture_score, mixture_learnware_list

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
        self.dbops.delete_learnware(id=id)

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

    def get_learnware_zip_path_by_ids(self, ids: Union[str, List[str]]) -> Union[Learnware, List[Learnware]]:
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

    def update_learnware_semantic_specification(self, learnware_id: str, semantic_spec: dict) -> bool:
        """Update Learnware semantic_spec"""

        # update database
        self.dbops.update_learnware_semantic_specification(learnware_id=learnware_id, semantic_spec=semantic_spec)
        # update file

        folder_path = self.learnware_folder_list[learnware_id]
        with open(os.path.join(folder_path, "semantic_specification.json"), "w") as f:
            json.dump(semantic_spec, f)
            pass
        # update zip
        zip_path = self.learnware_zip_list[learnware_id]
        utils.zip_learnware_folder(folder_path, zip_path)

        # update learnware
        self.learnware_list[learnware_id].update_semantic_spec(semantic_spec)
        pass

    def __len__(self):
        return len(self.learnware_list.keys())

    def _get_ids(self, top=None):
        if top is None:
            return list(self.learnware_list.keys())
        else:
            return list(self.learnware_list.keys())[:top]
