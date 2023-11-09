import torch
import unittest
import os
import copy
import joblib
import zipfile
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from shutil import copyfile, rmtree
from multiprocessing import Pool
from learnware.client import LearnwareClient
from sklearn.metrics import mean_squared_error

import learnware
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.specification import RKMETableSpecification, generate_rkme_spec
from learnware.reuse import HeteroMapTableReuser
from example_learnwares.config import input_shape_list, input_description_list, output_description_list, user_description_list

curr_root = os.path.dirname(os.path.abspath(__file__))

user_semantic = {
    "Data": {"Values": ["Table"], "Type": "Class"},
    "Task": {
        "Values": ["Regression"],
        "Type": "Class",
    },
    "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
    "Scenario": {"Values": ["Education"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
}

def check_learnware(learnware_name, dir_path=os.path.join(curr_root, "learnware_pool")):
    print(f"Checking Learnware: {learnware_name}")
    zip_file_path = os.path.join(dir_path, learnware_name)
    client = LearnwareClient()
    # if check_learnware doesn't raise an exception, return True, otherwise, return false
    try:
        client.check_learnware(zip_file_path)
        return True
    except Exception as e:
        print(f"Learnware {learnware_name} failed the check: {e}")
        return False


class TestMarket(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(2023)
        learnware.init()

    def _init_learnware_market(self):
        """initialize learnware market"""
        hetero_market = instantiate_learnware_market(market_id="hetero_toy", name="hetero", rebuild=True)
        return hetero_market


    def test_prepare_learnware_randomly(self, learnware_num=5):
        self.zip_path_list = []

        for i in range(learnware_num):
            dir_path = os.path.join(curr_root, "learnware_pool", "ridge_%d" % (i))
            os.makedirs(dir_path, exist_ok=True)

            print("Preparing Learnware: %d" % (i))

            example_learnware_idx=i%2
            input_dim=input_shape_list[example_learnware_idx]
            example_learnware_name="example_learnwares/example_learnware_%d" % (example_learnware_idx)

            X, y = make_regression(n_samples=5000, n_informative=15, n_features=input_dim, noise=0.1, random_state=42)

            clf=Ridge(alpha=1.0)
            clf.fit(X, y)

            joblib.dump(clf, os.path.join(dir_path, "ridge.pkl"))

            spec = generate_rkme_spec(X=X, gamma=0.1, cuda_idx=0)
            spec.save(os.path.join(dir_path, "stat.json"))

            init_file = os.path.join(dir_path, "__init__.py")
            copyfile(
                os.path.join(curr_root, example_learnware_name, "__init__.py"), init_file
            )  # cp example_init.py init_file

            yaml_file = os.path.join(dir_path, "learnware.yaml")
            copyfile(os.path.join(curr_root, example_learnware_name, "learnware.yaml"), yaml_file)  # cp example.yaml yaml_file

            env_file = os.path.join(dir_path, "requirements.txt")
            copyfile(os.path.join(curr_root, example_learnware_name, "requirements.txt"), env_file)

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

    def test_generated_learnwares(self):
        curr_root = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(curr_root, "learnware_pool")

        # Execute multi-process checking using Pool
        with Pool() as pool:
            results = pool.starmap(check_learnware, [(name, dir_path) for name in os.listdir(dir_path)])

        # Use an assert statement to ensure that all checks return True
        self.assertTrue(all(results), "Not all learnwares passed the check")

    def test_upload_delete_learnware(self, learnware_num=5, delete=True):
        hetero_market = self._init_learnware_market()
        self.test_prepare_learnware_randomly(learnware_num)
        self.learnware_num = learnware_num

        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == 0, f"The market should be empty!"

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = copy.deepcopy(user_semantic)
            semantic_spec["Name"]["Values"] = "learnware_%d" % (idx)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (idx)
            semantic_spec["Input"] = input_description_list[idx % 2]
            semantic_spec["Output"] = output_description_list[idx % 2]
            hetero_market.add_learnware(zip_path, semantic_spec)

        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"
        curr_inds = hetero_market.get_learnware_ids()
        print("Available ids After Uploading Learnwares:", curr_inds)
        assert len(curr_inds) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

        if delete:
            for learnware_id in curr_inds:
                hetero_market.delete_learnware(learnware_id)
                self.learnware_num -= 1
                assert len(hetero_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

            curr_inds = hetero_market.get_learnware_ids()
            print("Available ids After Deleting Learnwares:", curr_inds)
            assert len(curr_inds) == 0, f"The market should be empty!"

        return hetero_market
    
    def test_train_market_model(self, learnware_num=5):
        hetero_market = self._init_learnware_market()
        self.test_prepare_learnware_randomly(learnware_num)
        self.learnware_num = learnware_num
        hetero_market.learnware_organizer.reset(auto_update=True, auto_update_limit=learnware_num)

        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == 0, f"The market should be empty!"

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = copy.deepcopy(user_semantic)
            semantic_spec["Name"]["Values"] = "learnware_%d" % (idx)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (idx)
            semantic_spec["Input"] = input_description_list[idx % 2]
            semantic_spec["Output"] = output_description_list[idx % 2]
            hetero_market.add_learnware(zip_path, semantic_spec)

        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"
        curr_inds = hetero_market.get_learnware_ids()
        print("Available ids After Uploading Learnwares:", curr_inds)
        assert len(curr_inds) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

        # organizer=hetero_market.learnware_organizer
        # organizer.train(hetero_market.learnware_organizer.learnware_list.values())
        return hetero_market

    def test_search_semantics(self, learnware_num=5):
        hetero_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

        semantic_spec = copy.deepcopy(user_semantic)
        semantic_spec["Name"]["Values"] = f"learnware_{learnware_num - 1}"

        user_info = BaseUserInfo(semantic_spec=semantic_spec)
        _, single_learnware_list, _, _ = hetero_market.search_learnware(user_info)

        print("User info:", user_info.get_semantic_spec())
        print(f"Search result:")
        assert len(single_learnware_list) == 1, f"Exact semantic search failed!"
        for learnware in single_learnware_list:
            semantic_spec1 = learnware.get_specification().get_semantic_spec()
            print("Choose learnware:", learnware.id, semantic_spec1)
            assert semantic_spec1["Name"]["Values"] == semantic_spec["Name"]["Values"], f"Exact semantic search failed!"

        semantic_spec["Name"]["Values"] = "laernwaer"
        user_info = BaseUserInfo(semantic_spec=semantic_spec)
        _, single_learnware_list, _, _ = hetero_market.search_learnware(user_info)

        print("User info:", user_info.get_semantic_spec())
        print(f"Search result:")
        assert len(single_learnware_list) == self.learnware_num, f"Fuzzy semantic search failed!"
        for learnware in single_learnware_list:
            semantic_spec1 = learnware.get_specification().get_semantic_spec()
            print("Choose learnware:", learnware.id, semantic_spec1)

    def test_stat_search(self, learnware_num=5):
        hetero_market = self.test_train_market_model(learnware_num)
        print("Total Item:", len(hetero_market))

        # hetero test
        user_dim=15

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
            user_spec.load(os.path.join(unzip_dir, "stat.json"))
            z=user_spec.get_z()
            z=z[:,:user_dim]
            device=user_spec.device
            z=torch.tensor(z, device=device)
            user_spec.z=z

            semantic_spec = copy.deepcopy(user_semantic)
            semantic_spec["Input"]=copy.deepcopy(input_description_list[idx%2])
            semantic_spec["Input"]['Dimension']=user_dim
            # keep only the first user_dim descriptions
            semantic_spec["Input"]['Description']={key: semantic_spec["Input"]['Description'][str(key)] for key in range(user_dim)}

            user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})
            (
                sorted_score_list,
                single_learnware_list,
                mixture_score,
                mixture_learnware_list,
            ) = hetero_market.search_learnware(user_info)

            print(f"search result of user{idx}:")
            for score, learnware in zip(sorted_score_list, single_learnware_list):
                print(f"score: {score}, learnware_id: {learnware.id}")

        rmtree(test_folder)  # rm -r test_folder

        # homo test
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
            user_spec.load(os.path.join(unzip_dir, "stat.json"))
            user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec})
            (
                sorted_score_list,
                single_learnware_list,
                mixture_score,
                mixture_learnware_list,
            ) = hetero_market.search_learnware(user_info)

            target_spec_num=3 if idx%2==0 else 2
            assert len(single_learnware_list) == target_spec_num, f"Statistical search failed!"
            print(f"search result of user{idx}:")
            for score, learnware in zip(sorted_score_list, single_learnware_list):
                print(f"score: {score}, learnware_id: {learnware.id}")
            print(f"mixture_score: {mixture_score}\n")
            mixture_id = " ".join([learnware.id for learnware in mixture_learnware_list])
            print(f"mixture_learnware: {mixture_id}\n")

        rmtree(test_folder)  # rm -r test_folder

    def test_model_reuse(self, learnware_num=5):
        # generate toy regression problem
        X, y = make_regression(n_samples=5000, n_informative=10, n_features=15, noise=0.1, random_state=0)

        # generate rkme
        user_spec = generate_rkme_spec(X=X, gamma=0.1, cuda_idx=0)

        # generate specification
        semantic_spec = copy.deepcopy(user_semantic)
        semantic_spec["Input"] = user_description_list[0]
        user_info=BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})

        # learnware market search
        hetero_market = self.test_train_market_model(learnware_num)
        (
            sorted_score_list,
            single_learnware_list,
            mixture_score,
            mixture_learnware_list,
        ) = hetero_market.search_learnware(user_info)

        # print search results
        for score, learnware in zip(sorted_score_list, single_learnware_list):
            print(f"score: {score}, learnware_id: {learnware.id}")

        # model reuse
        reuser=HeteroMapTableReuser(single_learnware_list[0], task_type='regression')
        reuser.fit(user_spec)
        y_pred=reuser.predict(X)
        
        # calculate rmse
        rmse=mean_squared_error(y, y_pred, squared=False)
        print(f"rmse not finetune: {rmse}")

        # finetune
        reuser.finetune(X[:100], y[:100])
        y_pred=reuser.predict(X)
        rmse=mean_squared_error(y, y_pred, squared=False)
        print(f"rmse finetune: {rmse}")


def suite():
    _suite = unittest.TestSuite()
    # _suite.addTest(TestMarket("test_prepare_learnware_randomly"))
    # _suite.addTest(TestMarket("test_generated_learnwares"))
    # _suite.addTest(TestMarket("test_upload_delete_learnware"))
    # _suite.addTest(TestMarket("test_train_market_model"))
    # _suite.addTest(TestMarket("test_search_semantics"))
    # _suite.addTest(TestMarket("test_stat_search"))
    _suite.addTest(TestMarket("test_model_reuse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
