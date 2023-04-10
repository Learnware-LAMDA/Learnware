import os
import fire


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


if __name__ == "__main__":
    fire.Fire(LearnwareMarketWorkflow)
