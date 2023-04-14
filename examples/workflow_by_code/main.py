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

curr_root = os.path.dirname(os.path.abspath(__file__))

semantic_specs = [
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class",},
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Nature"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "learnware_1", "Type": "Name"},
    },
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class",},
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business", "Nature"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "learnware_2", "Type": "Name"},
    },
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class",},
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "learnware_3", "Type": "Name"},
    },
]

user_senmantic = {
    "Data": {"Values": ["Tabular"], "Type": "Class"},
    "Task": {"Values": ["Classification"], "Type": "Class",},
    "Device": {"Values": ["GPU"], "Type": "Tag"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "Description"},
    "Name": {"Values": "", "Type": "Name"},
}


class LearnwareMarketWorkflow:
    def _init_learnware_market(self):
        """initialize learnware market"""
        database_ops.clear_learnware_table()
        learnware.init()
        np.random.seed(2023)

    def prepare_learnware_randomly(self, learnware_num=10):
        self.zip_path_list = []
        for i in range(learnware_num):
            dir_path = os.path.join(curr_root, "learnware_pool", "svm_%d" % (i))
            os.makedirs(dir_path, exist_ok=True)

            print("Preparing Learnware: %d" % (i))
            data_X = np.random.randn(5000, 20) * i
            data_y = np.random.randn(5000)
            data_y = np.where(data_y > 0, 1, 0)

            clf = svm.SVC(kernel="linear")
            clf.fit(data_X, data_y)
            joblib.dump(clf, os.path.join(dir_path, "svm.pkl"))

            spec = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
            spec.save(os.path.join(dir_path, "svm.json"))

            init_file = os.path.join(dir_path, "__init__.py")
            os.system(f"cp example_init.py {init_file}")

            yaml_file = os.path.join(dir_path, "learnware.yaml")
            os.system(f"cp example.yaml {yaml_file}")

            zip_file = dir_path + ".zip"
            os.system(f"zip -q -r -j {zip_file} {dir_path}")
            os.system(f"rm -r {dir_path}")

            self.zip_path_list.append(zip_file)

    def test_upload_delete_learnware(self, learnware_num=5):
        self._init_learnware_market()
        self.prepare_learnware_randomly(learnware_num)

        easy_market = EasyMarket()
        print("Total Item:", len(easy_market))

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = semantic_specs[idx % 3]
            semantic_spec["Name"]["Values"] = "learnware_%d" % (idx)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (idx)
            easy_market.add_learnware(zip_path, semantic_spec)

        print("Total Item:", len(easy_market))
        curr_inds = easy_market._get_ids()
        print("Available ids:", curr_inds)

        easy_market.delete_learnware(curr_inds[0])
        easy_market.delete_learnware(curr_inds[1])
        curr_inds = easy_market._get_ids()
        print("Available ids:", curr_inds)

    def test_search_semantics(self, learnware_num=5):
        self._init_learnware_market()
        self.prepare_learnware_randomly(learnware_num)

        easy_market = EasyMarket()
        print("Total Item:", len(easy_market))

        test_folder = os.path.join(curr_root, "test_semantics")

        idx, zip_path = 1, self.zip_path_list[1]
        unzip_dir = os.path.join(test_folder, f"{idx}")
        os.makedirs(unzip_dir, exist_ok=True)
        os.system(f"unzip -o -q {zip_path} -d {unzip_dir}")

        user_info = BaseUserInfo(id="user_0", semantic_spec=user_senmantic)
        _, single_learnware_list, _ = easy_market.search_learnware(user_info)

        print("User info:", user_info.get_semantic_spec())
        print(f"search result of user{idx}:")
        for learnware in single_learnware_list:
            print("Choose learnware:", learnware.id, learnware.get_specification().get_semantic_spec())

        os.system(f"rm -r {test_folder}")

    def test_stat_search(self, learnware_num=5):
        self._init_learnware_market()
        self.prepare_learnware_randomly(learnware_num)

        print(self.zip_path_list)
        easy_market = EasyMarket()
        print("Total Item:", len(easy_market))

        test_folder = os.path.join(curr_root, "test_stat")

        for idx, zip_path in enumerate(self.zip_path_list):
            unzip_dir = os.path.join(test_folder, f"{idx}")
            os.makedirs(unzip_dir, exist_ok=True)
            os.system(f"unzip -o -q {zip_path} -d {unzip_dir}")

            user_spec = specification.rkme.RKMEStatSpecification()
            user_spec.load(os.path.join(unzip_dir, "svm.json"))
            user_info = BaseUserInfo(
                id="user_0", semantic_spec=user_senmantic, stat_info={"RKMEStatSpecification": user_spec}
            )
            sorted_score_list, single_learnware_list, mixture_learnware_list = easy_market.search_learnware(user_info)

            print(f"search result of user{idx}:")
            for score, learnware in zip(sorted_score_list, single_learnware_list):
                print(f"score: {score}, learnware_id: {learnware.id}")
            mixture_id = " ".join([learnware.id for learnware in mixture_learnware_list])
            print(f"mixture_learnware: {mixture_id}\n")

        os.system(f"rm -r {test_folder}")


if __name__ == "__main__":
    fire.Fire(LearnwareMarketWorkflow)
