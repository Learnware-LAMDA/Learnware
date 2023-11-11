import os
import fire
import copy
import joblib
import zipfile
import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from shutil import copyfile, rmtree

import learnware
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser
from learnware.specification import generate_rkme_table_spec, RKMETableSpecification

curr_root = os.path.dirname(os.path.abspath(__file__))

user_semantic = {
    "Data": {"Values": ["Table"], "Type": "Class"},
    "Task": {
        "Values": ["Classification"],
        "Type": "Class",
    },
    "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
    "Scenario": {"Values": ["Education"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
}


class LearnwareMarketWorkflow:
    def _init_learnware_market(self):
        """initialize learnware market"""
        learnware.init()
        np.random.seed(2023)
        easy_market = instantiate_learnware_market(market_id="sklearn_digits", name="easy", rebuild=True)
        return easy_market

    def prepare_learnware_randomly(self, learnware_num=5):
        self.zip_path_list = []
        X, y = load_digits(return_X_y=True)

        for i in range(learnware_num):
            dir_path = os.path.join(curr_root, "learnware_pool", "svm_%d" % (i))
            os.makedirs(dir_path, exist_ok=True)

            print("Preparing Learnware: %d" % (i))

            data_X, _, data_y, _ = train_test_split(X, y, test_size=0.3, shuffle=True)
            clf = svm.SVC(kernel="linear", probability=True)
            clf.fit(data_X, data_y)

            joblib.dump(clf, os.path.join(dir_path, "svm.pkl"))

            spec = generate_rkme_table_spec(X=data_X, gamma=0.1, cuda_idx=0)
            spec.save(os.path.join(dir_path, "svm.json"))

            init_file = os.path.join(dir_path, "__init__.py")
            copyfile(
                os.path.join(curr_root, "learnware_example/example_init.py"), init_file
            )  # cp example_init.py init_file

            yaml_file = os.path.join(dir_path, "learnware.yaml")
            copyfile(os.path.join(curr_root, "learnware_example/example.yaml"), yaml_file)  # cp example.yaml yaml_file

            zip_file = dir_path + ".zip"
            # zip -q -r -j zip_file dir_path
            with zipfile.ZipFile(zip_file, "w") as zip_obj:
                for foldername, subfolders, filenames in os.walk(dir_path):
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        zip_info = zipfile.ZipInfo(filename)
                        zip_info.compress_type = zipfile.ZIP_STORED
                        with open(file_path, "rb") as file:
                            zip_obj.writestr(zip_info, file.read())

            rmtree(dir_path)  # rm -r dir_path

            self.zip_path_list.append(zip_file)

    def test_upload_delete_learnware(self, learnware_num=5, delete=False):
        easy_market = self._init_learnware_market()
        self.prepare_learnware_randomly(learnware_num)

        print("Total Item:", len(easy_market))

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = copy.deepcopy(user_semantic)
            semantic_spec["Name"]["Values"] = "learnware_%d" % (idx)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (idx)
            easy_market.add_learnware(zip_path, semantic_spec)

        print("Total Item:", len(easy_market))
        curr_inds = easy_market.get_learnware_ids()
        print("Available ids After Uploading Learnwares:", curr_inds)

        if delete:
            for learnware_id in curr_inds:
                easy_market.delete_learnware(learnware_id)
            curr_inds = easy_market.get_learnware_ids()
            print("Available ids After Deleting Learnwares:", curr_inds)

        return easy_market

    def test_search_semantics(self, learnware_num=5):
        easy_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(easy_market))

        test_folder = os.path.join(curr_root, "test_semantics")

        # unzip -o -q zip_path -d unzip_dir
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.makedirs(test_folder, exist_ok=True)

        with zipfile.ZipFile(self.zip_path_list[0], "r") as zip_obj:
            zip_obj.extractall(path=test_folder)

        semantic_spec = copy.deepcopy(user_semantic)
        semantic_spec["Name"]["Values"] = f"learnware_{learnware_num - 1}"
        semantic_spec["Description"]["Values"] = f"test_learnware_number_{learnware_num - 1}"

        user_info = BaseUserInfo(semantic_spec=semantic_spec)
        _, single_learnware_list, _, _ = easy_market.search_learnware(user_info)

        print("User info:", user_info.get_semantic_spec())
        print(f"Search result:")
        for learnware in single_learnware_list:
            print("Choose learnware:", learnware.id, learnware.get_specification().get_semantic_spec())

        rmtree(test_folder)  # rm -r test_folder

    def test_stat_search(self, learnware_num=5):
        easy_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(easy_market))

        test_folder = os.path.join(curr_root, "test_stat")

        for idx, zip_path in enumerate(self.zip_path_list):
            unzip_dir = os.path.join(test_folder, f"{idx}")

            # unzip -o -q zip_path -d unzip_dir
            if os.path.exists(unzip_dir):
                rmtree(unzip_dir)
            os.makedirs(unzip_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_obj:
                zip_obj.extractall(path=unzip_dir)

            user_spec = RKMETableSpecification()
            user_spec.load(os.path.join(unzip_dir, "svm.json"))
            user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec})
            (
                sorted_score_list,
                single_learnware_list,
                mixture_score,
                mixture_learnware_list,
            ) = easy_market.search_learnware(user_info)

            print(f"search result of user{idx}:")
            for score, learnware in zip(sorted_score_list, single_learnware_list):
                print(f"score: {score}, learnware_id: {learnware.id}")
            print(f"mixture_score: {mixture_score}\n")
            mixture_id = " ".join([learnware.id for learnware in mixture_learnware_list])
            print(f"mixture_learnware: {mixture_id}\n")

        rmtree(test_folder)  # rm -r test_folder

    def test_learnware_reuse(self, learnware_num=5):
        easy_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(easy_market))

        X, y = load_digits(return_X_y=True)
        _, data_X, _, data_y = train_test_split(X, y, test_size=0.3, shuffle=True)

        stat_spec = generate_rkme_table_spec(X=data_X, gamma=0.1, cuda_idx=0)
        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": stat_spec})

        _, _, _, mixture_learnware_list = easy_market.search_learnware(user_info)

        # print("Mixture Learnware:", mixture_learnware_list)

        # Based on user information, the learnware market returns a list of learnwares (learnware_list)
        # Use jobselector reuser to reuse the searched learnwares to make prediction
        reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
        job_selector_predict_y = reuse_job_selector.predict(user_data=data_X)

        # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list)
        ensemble_predict_y = reuse_ensemble.predict(user_data=data_X)

        print("Job Selector Acc:", np.sum(np.argmax(job_selector_predict_y, axis=1) == data_y) / len(data_y))
        print("Averaging Selector Acc:", np.sum(np.argmax(ensemble_predict_y, axis=1) == data_y) / len(data_y))


if __name__ == "__main__":
    fire.Fire(LearnwareMarketWorkflow)
