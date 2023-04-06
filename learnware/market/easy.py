import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Any, List, Union, Dict

from .base import BaseMarket, BaseUserInfo
from ..learnware import Learnware
from ..specification import RKMEStatSpecification, Specification
from .database_ops import load_market_from_db, add_learnware_to_db, delete_learnware_from_db


class EasyMarket(BaseMarket):
    def __init__(self):
        """Initializing an empty market"""
        self.learnware_list = {}  # id: Learnware
        self.count = 0
        self.semantic_spec_list = self._init_semantic_spec_list()

    def _init_semantic_spec_list(self):
        # TODO: Load from json
        return {
            "Data": {
                "Values": ["Tabular", "Image", "Video", "Text", "Audio"],
                "Type": "Class",  # Choose only one class
            },
            "Task": {
                "Values": [
                    "Classification",
                    "Regression",
                    "Clustering",
                    "Feature Extraction",
                    "Generation",
                    "Segmentation",
                    "Object Detection",
                ],
                "Type": "Class",  # Choose only one class
            },
            "Device": {
                "Values": ["CPU", "GPU"],
                "Type": "Tag",  # Choose one or more tags
            },
            "Scenario": {
                "Values": [
                    "Business",
                    "Financial",
                    "Health",
                    "Politics",
                    "Computer",
                    "Internet",
                    "Traffic",
                    "Nature",
                    "Fashion",
                    "Industry",
                    "Agriculture",
                    "Education",
                    "Entertainment",
                    "Architecture",
                ],
                "Type": "Tag",  # Choose one or more tags
            },
            "Description": {
                "Values": str,
                "Type": "Description",
            },
        }

    def reload_market(self) -> bool:
        self.learnware_list, self.count = load_market_from_db()

    def add_learnware(
        self, learnware_name: str, model_path: str, stat_spec_path: str, semantic_spec: dict, desc: str
    ) -> Tuple[str, bool]:
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
        if (not os.path.exists(model_path)) or (not os.path.exists(stat_spec_path)):
            raise FileNotFoundError("Model or Stat_spec NOT Found.")

        id = "%08d" % (self.count)
        rkme_stat_spec = RKMEStatSpecification()
        rkme_stat_spec.load(stat_spec_path)
        stat_spec = {"RKME": rkme_stat_spec}
        specification = Specification(semantic_spec=semantic_spec, stat_spec=stat_spec)
        # specification.update_stat_spec("RKME", rkme_stat_spec)
        model_dict = {"model_path": model_path, "class_name": "BaseModel"}
        new_learnware = Learnware(id=id, name=learnware_name, model=model_dict, specification=specification)
        self.learnware_list[id] = new_learnware
        self.count += 1

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

        term1 = user_rkme.eval_Phi(user_rkme)
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
        sorted_dist_list, sorted_learnware_list = (list(t) for t in zip(*sorted(zip(mmd_dist_list, learnware_list))))

        return sorted_dist_list, sorted_learnware_list

    def search_learnware(self, user_info: BaseUserInfo) -> Tuple[Any, List[Learnware]]:
        def search_by_semantic_spec():
            def match_semantic_spec(semantic_spec1, semantic_spec2):
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
            # TODO: self.learnware_list is a dict. Bug need to be fixed!
            for learnware in self.learnware_list:
                learnware_semantic_spec = learnware.get_specification().get_semantic_spec()
                user_semantic_spec = user_info.get_semantic_spec()
                if match_semantic_spec(learnware_semantic_spec, user_semantic_spec):
                    match_learnwares.append(learnware)
            return match_learnwares

        match_learnwares = search_by_semantic_spec()
        # return match_learnwares
        # TODO:

    def delete_learnware(self, id: str) -> bool:
        if not id in self.learnware_list:
            raise Exception("Learnware id:{} NOT Found!".format(id))

        self.learnware_list.pop(id)
        return True

    def get_semantic_spec_list(self) -> dict:
        return self.semantic_spec_list
