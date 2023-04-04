import os
import numpy as np
import pandas as pd
from typing import Tuple, Any, List, Union, Dict

from .base import BaseMarket, BaseUserInfo
from ..learnware import Learnware
from ..specification import RKMEStatSpecification

class SerialUserInfo(BaseUserInfo):
    def __init__(self, id: str, semantic_spec: dict = dict(), stat_info: dict = dict()):
        """Initializing user information

        Parameters
        ----------
        id : str
            user id
        semantic_spec : dict, optional
            semantic_spec selected by user, by default dict()
        stat_info : dict, optional
            statistical information uploaded by user, by default dict()
        """
        self.id = id
        self.semantic_spec = semantic_spec
        self.stat_info = stat_info

    def get_semantic_spec(self) -> dict:
        """Return user semantic specifications

        Returns
        -------
        dict
            user semantic specifications
        """
        return self.semantic_spec

    def get_stat_info(self, name: str):
        return self.stat_info.get(name, None)

class SerialMarket(BaseMarket):
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

    def reload_market(self, market_path: str, semantic_spec_list_path: str) -> bool:
        raise NotImplementedError("reload market is Not Implemented")

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
        stat_spec = RKMEStatSpecification()
        stat_spec_path.load(stat_spec_path)

        return str(self.count), True

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
            for learnware in self.learnware_list:
                learnware_semantic_spec = learnware.get_specification().get_semantic_spec()
                user_semantic_spec = user_info.get_semantic_spec()
                if match_semantic_spec(learnware_semantic_spec, user_semantic_spec):
                    match_learnwares.append(learnware)
            return match_learnwares

        match_learnwares = search_by_semantic_spec()

    def delete_learnware(self, id: str) -> bool:
        if not id in self.learnware_list:
            raise Exception("Learnware id:{} NOT Found!".format(id))

        self.learnware_list.pop(id)
        return True

    def get_semantic_spec_list(self) -> dict:
        return self.semantic_spec_list
