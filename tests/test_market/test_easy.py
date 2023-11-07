import sys
import unittest
import os
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
from learnware.specification import RKMETableSpecification, generate_rkme_spec
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser

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


class TestMarket(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(2023)
        learnware.init()

    def _init_learnware_market(self):
        """initialize learnware market"""
        easy_market = instantiate_learnware_market(market_id="sklearn_digits_easy", name="easy", rebuild=True)
        return easy_market

    def test_prepare_learnware_randomly(self, learnware_num=5):
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

            spec = generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
            spec.save(os.path.join(dir_path, "svm.json"))

            init_file = os.path.join(dir_path, "__init__.py")
            copyfile(
                os.path.join(curr_root, "learnware_example/example_init.py"), init_file
            )  # cp example_init.py init_file

            yaml_file = os.path.join(dir_path, "learnware.yaml")
            copyfile(os.path.join(curr_root, "learnware_example/example.yaml"), yaml_file)  # cp example.yaml yaml_file

            env_file = os.path.join(dir_path, "environment.yaml")
            copyfile(os.path.join(curr_root, "learnware_example/environment.yaml"), env_file)

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

    def test_upload_delete_learnware(self, learnware_num=5, delete=True):
        easy_market = self._init_learnware_market()
        self.test_prepare_learnware_randomly(learnware_num)
        self.learnware_num = learnware_num

        print("Total Item:", len(easy_market))
        assert len(easy_market) == 0, f"The market should be empty!"

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = copy.deepcopy(user_semantic)
            semantic_spec["Name"]["Values"] = "learnware_%d" % (idx)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (idx)
            semantic_spec["Input"] = {"Dimension": 64, "Description": {f"{i}": f"The value in the grid {i // 8}{i % 8} of the image of hand-written digit." for i in range(64)}}
            semantic_spec["Output"] = {"Dimension": 10, "Description": {f"{i}": "The probability for each digit for 0 to 9." for i in range(10)}}
            easy_market.add_learnware(zip_path, semantic_spec)

        print("Total Item:", len(easy_market))
        assert len(easy_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"
        curr_inds = easy_market.get_learnware_ids()
        print("Available ids After Uploading Learnwares:", curr_inds)
        assert len(curr_inds) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

        if delete:
            for learnware_id in curr_inds:
                easy_market.delete_learnware(learnware_id)
                self.learnware_num -= 1
                assert len(easy_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

            curr_inds = easy_market.get_learnware_ids()
            print("Available ids After Deleting Learnwares:", curr_inds)
            assert len(curr_inds) == 0, f"The market should be empty!"

        return easy_market

    def test_search_semantics(self, learnware_num=5):
        easy_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(easy_market))
        assert len(easy_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"
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

            assert len(single_learnware_list) == self.learnware_num, f"Statistical search failed!"
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
        train_X, data_X, train_y, data_y = train_test_split(X, y, test_size=0.3, shuffle=True)

        stat_spec = generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": stat_spec})

        _, _, _, mixture_learnware_list = easy_market.search_learnware(user_info)

        # Based on user information, the learnware market returns a list of learnwares (learnware_list)
        # Use jobselector reuser to reuse the searched learnwares to make prediction
        reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list)
        job_selector_predict_y = reuse_job_selector.predict(user_data=data_X)

        # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote_by_prob")
        ensemble_predict_y = reuse_ensemble.predict(user_data=data_X)

        # Use ensemble pruning reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = EnsemblePruningReuser(learnware_list=mixture_learnware_list, mode="classification")
        reuse_ensemble.fit(train_X[-200:], train_y[-200:])
        ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=data_X)

        print("Job Selector Acc:", np.sum(np.argmax(job_selector_predict_y, axis=1) == data_y) / len(data_y))
        print("Averaging Reuser Acc:", np.sum(np.argmax(ensemble_predict_y, axis=1) == data_y) / len(data_y))
        print("Ensemble Pruning Reuser Acc:", np.sum(ensemble_pruning_predict_y == data_y) / len(data_y))

def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(TestMarket("test_prepare_learnware_randomly"))
    _suite.addTest(TestMarket("test_upload_delete_learnware"))
    _suite.addTest(TestMarket("test_search_semantics"))
    _suite.addTest(TestMarket("test_stat_search"))
    _suite.addTest(TestMarket("test_learnware_reuse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
