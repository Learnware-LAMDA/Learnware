import os
import fire
import zipfile
import time
import numpy as np
from tqdm import tqdm
from shutil import copyfile, rmtree

import learnware
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser
from learnware.specification import generate_rkme_table_spec
from pfs import Dataloader
from learnware.logger import get_module_logger

logger = get_module_logger("pfs_test", level="INFO")

output_description = {
    "Dimension": 1,
    "Description": {},
}

input_description = {
    "Dimension": 31,
    "Description": {},
}

semantic_specs = [
    {
        "Data": {"Values": ["Table"], "Type": "Class"},
        "Task": {"Values": ["Regression"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "learnware_1", "Type": "String"},
        "Input": input_description,
        "Output": output_description,
        "License": {"Values": ["MIT"], "Type": "Class"},
    }
]

user_semantic = {
    "Data": {"Values": ["Table"], "Type": "Class"},
    "Task": {"Values": ["Regression"], "Type": "Class"},
    "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
    "Input": input_description,
    "Output": output_description,
    "License": {"Values": ["MIT"], "Type": "Class"},
}


class PFSDatasetWorkflow:
    def _init_pfs_dataset(self):
        pfs = Dataloader()
        pfs.regenerate_data()

        algo_list = ["ridge"]  # "ridge", "lgb"
        for algo in algo_list:
            pfs.set_algo(algo)
            pfs.retrain_models()

    def _init_learnware_market(self):
        """initialize learnware market"""
        learnware.init()
        easy_market = instantiate_learnware_market(market_id="pfs", name="easy", rebuild=True)
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

    def prepare_learnware(self, regenerate_flag=False):
        if regenerate_flag:
            self._init_pfs_dataset()

        pfs = Dataloader()
        idx_list = pfs.get_idx_list()
        algo_list = ["ridge"]  # ["ridge", "lgb"]

        curr_root = os.path.dirname(os.path.abspath(__file__))
        curr_root = os.path.join(curr_root, "learnware_pool")
        os.makedirs(curr_root, exist_ok=True)

        for idx in tqdm(idx_list):
            train_x, train_y, test_x, test_y = pfs.get_idx_data(idx)
            st = time.time()
            spec = generate_rkme_table_spec(X=train_x, gamma=0.1, cuda_idx=0)
            ed = time.time()
            logger.info("Stat spec generated in %.3f s" % (ed - st))

            for algo in algo_list:
                pfs.set_algo(algo)
                dir_path = os.path.join(curr_root, f"{algo}_{idx}")
                os.makedirs(dir_path, exist_ok=True)

                spec_path = os.path.join(dir_path, "rkme.json")
                spec.save(spec_path)

                model_path = pfs.get_model_path(idx)
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

        easy_market = instantiate_learnware_market(market_id="pfs", name="easy")
        print("Total Item:", len(easy_market))

        pfs = Dataloader()
        idx_list = pfs.get_idx_list()
        os.makedirs("./user_spec", exist_ok=True)
        single_score_list = []
        random_score_list = []
        job_selector_score_list = []
        ensemble_score_list = []
        improve_list = []

        for idx in idx_list:
            train_x, train_y, test_x, test_y = pfs.get_idx_data(idx)
            user_spec = generate_rkme_table_spec(X=test_x, gamma=0.1, cuda_idx=0)
            user_spec_path = f"./user_spec/user_{idx}.json"
            user_spec.save(user_spec_path)

            user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec})
            search_result = easy_market.search_learnware(user_info)
            single_result = search_result.get_single_results()
            multiple_result = search_result.get_multiple_results()

            print(f"search result of user{idx}:")
            print(
                f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
            )
            loss_list = []
            for single_item in single_result:
                pred_y = single_item.learnware.predict(test_x)
                loss_list.append(pfs.score(test_y, pred_y))
            print(
                f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, loss: {loss_list[0]}, random: {np.mean(loss_list)}"
            )

            if len(multiple_result) > 0:
                mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                print(f"mixture_score: {multiple_result[0].score}, mixture_learnware: {mixture_id}")
                mixture_learnware_list = multiple_result[0].learnwares
            else:
                mixture_learnware_list = [single_result[0].learnware]

            reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list, use_herding=False)
            job_selector_predict_y = reuse_job_selector.predict(user_data=test_x)
            job_selector_score = pfs.score(test_y, job_selector_predict_y)
            print(f"mixture reuse loss (job selector): {job_selector_score}")

            reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list)
            ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
            ensemble_score = pfs.score(test_y, ensemble_predict_y)
            print(f"mixture reuse loss (ensemble): {ensemble_score}\n")

            single_score_list.append(loss_list[0])
            random_score_list.append(np.mean(loss_list))
            job_selector_score_list.append(job_selector_score)
            ensemble_score_list.append(ensemble_score)
            improve_list.append((np.mean(loss_list) - loss_list[0]) / np.mean(loss_list))

        logger.info("Single search score %.3f +/- %.3f" % (np.mean(single_score_list), np.std(single_score_list)))
        logger.info("Random search score: %.3f +/- %.3f" % (np.mean(random_score_list), np.std(random_score_list)))
        logger.info("Average score improvement: %.3f" % (np.mean(improve_list)))
        logger.info(
            "Job selector score: %.3f +/- %.3f" % (np.mean(job_selector_score_list), np.std(job_selector_score_list))
        )
        logger.info(
            "Average ensemble score: %.3f +/- %.3f" % (np.mean(ensemble_score_list), np.std(ensemble_score_list))
        )


if __name__ == "__main__":
    fire.Fire(PFSDatasetWorkflow)
