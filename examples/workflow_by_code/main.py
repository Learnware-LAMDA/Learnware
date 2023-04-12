import fire
import os
import joblib
import numpy as np
import learnware

from sklearn import svm
from learnware.market import EasyMarket, BaseUserInfo
from learnware.market import database_ops
from learnware.learnware import Learnware
import learnware.specification as specification
from learnware.utils import get_module_by_module_path


class LearnwareMarketWorkflow:
    curr_root = os.path.dirname(os.path.abspath(__file__))

    semantic_specs = [
        {
            "Data": {"Values": ["Tabular"], "Type": "Class"},
            "Task": {
                "Values": ["Classification"],
                "Type": "Class",
            },
            "Device": {"Values": ["GPU"], "Type": "Tag"},
            "Scenario": {"Values": ["Nature"], "Type": "Tag"},
            "Description": {"Values": "", "Type": "Description"},
            "Name": {"Values": "learnware_1", "Type": "Name"},
        },
        {
            "Data": {"Values": ["Tabular"], "Type": "Class"},
            "Task": {
                "Values": ["Classification"],
                "Type": "Class",
            },
            "Device": {"Values": ["GPU"], "Type": "Tag"},
            "Scenario": {"Values": ["Business", "Nature"], "Type": "Tag"},
            "Description": {"Values": "", "Type": "Description"},
            "Name": {"Values": "learnware_2", "Type": "Name"},
        },
        {
            "Data": {"Values": ["Tabular"], "Type": "Class"},
            "Task": {
                "Values": ["Classification"],
                "Type": "Class",
            },
            "Device": {"Values": ["GPU"], "Type": "Tag"},
            "Scenario": {"Values": ["Business"], "Type": "Tag"},
            "Description": {"Values": "", "Type": "Description"},
            "Name": {"Values": "learnware_3", "Type": "Name"},
        },
    ]

    user_senmantic = {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {
            "Values": ["Classification"],
            "Type": "Class",
        },
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "", "Type": "Name"},
    }

    def _init_learnware_market(self):
        """initialize learnware market"""
        database_ops.clear_learnware_table()
        learnware.init()

        self.learnware_market = EasyMarket()

    def _generate_learnware_randomly(self):
        pass

    # def _


if __name__ == "__main__":
    fire.Fire(LearnwareMarketWorkflow)
