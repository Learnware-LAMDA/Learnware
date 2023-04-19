import os
import fire
import zipfile
from tqdm import tqdm
from shutil import copyfile, rmtree

import learnware
from learnware.market import EasyMarket, BaseUserInfo
from learnware.market import database_ops
from learnware.learnware import Learnware, JobSelectorReuser
import learnware.specification as specification
from m5 import DataLoader


semantic_specs = [
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "learnware_1", "Type": "String"},
    }
]

user_senmantic = {
    "Data": {"Values": ["Tabular"], "Type": "Class"},
    "Task": {"Values": ["Classification"], "Type": "Class"},
    "Device": {"Values": ["GPU"], "Type": "Tag"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
}


class M5DatasetWorkflow:
    def _init_m5_dataset(self):
        m5 = DataLoader()
        m5.regenerate_data()

        algo_list = ["ridge", "lgb"]
        for algo in algo_list:
            m5.set_algo(algo)
            m5.retrain_models()

    def _init_learnware_market(self):
        """initialize learnware market"""
        database_ops.clear_learnware_table()
        learnware.init()

        easy_market = EasyMarket()
        print("Total Item:", len(easy_market))

        zip_path_list = []
        curr_root = os.path.dirname(os.path.abspath(__file__))
        curr_root = os.path.join(curr_root, "learnware_pool")
        for zip_path in os.listdir(curr_root):
            zip_path_list.append(os.path.join(curr_root, zip_path))

        for idx, zip_path in enumerate(zip_path_list):
            semantic_spec = semantic_specs[0]
            semantic_spec["Name"]["Values"] = "learnware_%d" % (idx)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (idx)
            easy_market.add_learnware(zip_path, semantic_spec)

        print("Total Item:", len(easy_market))
        curr_inds = easy_market._get_ids()
        print("Available ids:", curr_inds)

    def prepare_learnware(self, regenerate_flag=False):
        if regenerate_flag:
            self._init_m5_dataset()

        m5 = DataLoader()
        idx_list = m5.get_idx_list()
        algo_list = ["lgb"]  # algo_list = ["ridge", "lgb"]

        curr_root = os.path.dirname(os.path.abspath(__file__))
        curr_root = os.path.join(curr_root, "learnware_pool")
        os.makedirs(curr_root, exist_ok=True)

        for idx in tqdm(idx_list):
            train_x, train_y, test_x, test_y = m5.get_idx_data(idx)
            spec = specification.utils.generate_rkme_spec(X=train_x, gamma=0.1, cuda_idx=0)

            for algo in algo_list:
                m5.set_algo(algo)
                dir_path = os.path.join(curr_root, f"{algo}_{idx}")
                os.makedirs(dir_path, exist_ok=True)

                spec_path = os.path.join(dir_path, "rkme.json")
                spec.save(spec_path)

                model_path = m5.get_model_path(idx)
                model_file = os.path.join(dir_path, "model.out")
                copyfile(model_path, model_file)

                init_file = os.path.join(dir_path, "__init__.py")
                copyfile("example_init.py", init_file)

                yaml_file = os.path.join(dir_path, "learnware.yaml")
                copyfile("example.yaml", yaml_file)

                zip_file = dir_path + ".zip"
                with zipfile.ZipFile(zip_file, "w") as zip_obj:
                    for foldername, subfolders, filenames in os.walk(dir_path):
                        for filename in filenames:
                            file_path = os.path.join(foldername, filename)
                            zip_info = zipfile.ZipInfo(filename)
                            zip_info.compress_type = zipfile.ZIP_STORED
                            with open(file_path, "rb") as file:
                                zip_obj.writestr(zip_info, file.read())

                rmtree(dir_path)

    def test(self, regenerate_flag=False):
        self.prepare_learnware(regenerate_flag)
        self._init_learnware_market()

        easy_market = EasyMarket()
        print("Total Item:", len(easy_market))

        m5 = DataLoader()
        idx_list = m5.get_idx_list()

        for idx in idx_list:
            train_x, train_y, test_x, test_y = m5.get_idx_data(idx)
            user_spec = specification.utils.generate_rkme_spec(X=test_x, gamma=0.1, cuda_idx=0)

            user_info = BaseUserInfo(
                id=f"user_{idx}", semantic_spec=user_senmantic, stat_info={"RKMEStatSpecification": user_spec}
            )
            sorted_score_list, single_learnware_list, mixture_learnware_list = easy_market.search_learnware(user_info)

            print(f"search result of user{idx}:")
            print(
                f"single model num: {len(sorted_score_list)}, max_score: {sorted_score_list[0]}, min_score: {sorted_score_list[-1]}"
            )
            for score, learnware in zip(sorted_score_list, single_learnware_list):
                pred_y = learnware.predict(test_x)
                loss = m5.score(test_y, pred_y)
                print(f"score: {score}, learnware_id: {learnware.id}, loss: {loss}")

            mixture_id = " ".join([learnware.id for learnware in mixture_learnware_list])
            print(f"mixture_learnware: {mixture_id}\n")

            reuse_baseline = JobSelectorReuser(learnware_list=mixture_learnware_list)
            reuse_predict = reuse_baseline.predict(user_data=test_x)
            reuse_score = m5.score(test_y, reuse_predict)
            print(f"mixture reuse loss: {reuse_score}\n")


if __name__ == "__main__":
    fire.Fire(M5DatasetWorkflow)
