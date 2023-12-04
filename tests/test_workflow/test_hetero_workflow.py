import torch
import pickle
import unittest
import os
import logging
import tempfile
import zipfile
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from shutil import copyfile, rmtree
from sklearn.metrics import mean_squared_error

import learnware
learnware.init(logging_level=logging.WARNING)

from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.specification import RKMETableSpecification, generate_rkme_table_spec, generate_semantic_spec
from learnware.reuse import HeteroMapAlignLearnware, AveragingReuser, EnsemblePruningReuser
from learnware.tests.templates import LearnwareTemplate, PickleModelTemplate, StatSpecTemplate

from hetero_config import input_shape_list, input_description_list, output_description_list, user_description_list


curr_root = os.path.dirname(os.path.abspath(__file__))

class TestHeteroWorkflow(unittest.TestCase):
    universal_semantic_config = {
        "data_type": "Table",
        "task_type": "Regression",
        "library_type": "Scikit-learn",
        "scenarios": "Education",
        "license": "MIT",
    }

    def _init_learnware_market(self, organizer_kwargs=None):
        """initialize learnware market"""
        hetero_market = instantiate_learnware_market(
            market_id="hetero_toy", name="hetero", rebuild=True, organizer_kwargs=organizer_kwargs
        )
        return hetero_market

    def test_prepare_learnware_randomly(self, learnware_num=5):
        self.zip_path_list = []

        for i in range(learnware_num):
            learnware_pool_dirpath = os.path.join(curr_root, "learnware_pool_hetero")
            os.makedirs(learnware_pool_dirpath, exist_ok=True)
            learnware_zippath = os.path.join(learnware_pool_dirpath, "ridge_%d.zip" % (i))
            
            print("Preparing Learnware: %d" % (i))

            X, y = make_regression(n_samples=5000, n_informative=15, n_features=input_shape_list[i % 2], noise=0.1, random_state=42)
            clf = Ridge(alpha=1.0)
            clf.fit(X, y)
            pickle_filepath = os.path.join(learnware_pool_dirpath, "ridge.pkl")
            with open(pickle_filepath, "wb") as fout:
                pickle.dump(clf, fout)

            spec = generate_rkme_table_spec(X=X, gamma=0.1)
            spec_filepath = os.path.join(learnware_pool_dirpath, "stat_spec.json")
            spec.save(spec_filepath)

            LearnwareTemplate.generate_learnware_zipfile(
                learnware_zippath=learnware_zippath,
                model_template=PickleModelTemplate(pickle_filepath=pickle_filepath, model_kwargs={"input_shape":(input_shape_list[i % 2],), "output_shape": (1,)}),
                stat_spec_template=StatSpecTemplate(filepath=spec_filepath, type="RKMETableSpecification"),
                requirements=["scikit-learn==0.22"],
            )
            
            self.zip_path_list.append(learnware_zippath)

    
    def _upload_delete_learnware(self, hetero_market, learnware_num, delete):
        self.test_prepare_learnware_randomly(learnware_num)
        self.learnware_num = learnware_num

        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == 0, f"The market should be empty!"

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = generate_semantic_spec(
                name=f"learnware_{idx}",
                description=f"test_learnware_number_{idx}",
                input_description=input_description_list[idx % 2],
                output_description=output_description_list[idx % 2],
                **self.universal_semantic_config
            )
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
                assert (
                    len(hetero_market) == self.learnware_num
                ), f"The number of learnwares must be {self.learnware_num}!"

            curr_inds = hetero_market.get_learnware_ids()
            print("Available ids After Deleting Learnwares:", curr_inds)
            assert len(curr_inds) == 0, f"The market should be empty!"

        return hetero_market
    
    def test_upload_delete_learnware(self, learnware_num=5, delete=True):
        hetero_market = self._init_learnware_market()
        return self._upload_delete_learnware(hetero_market, learnware_num, delete)

    def test_train_market_model(self, learnware_num=5, delete=False):
        hetero_market = self._init_learnware_market(
            organizer_kwargs={"auto_update": True, "auto_update_limit": learnware_num}
        )
        hetero_market = self._upload_delete_learnware(hetero_market, learnware_num, delete)
        # organizer=hetero_market.learnware_organizer
        # organizer.train(hetero_market.learnware_organizer.learnware_list.values())
        return hetero_market

    def test_search_semantics(self, learnware_num=5):
        hetero_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(hetero_market))
        assert len(hetero_market) == self.learnware_num, f"The number of learnwares must be {self.learnware_num}!"

        semantic_spec = generate_semantic_spec(
            name=f"learnware_{learnware_num - 1}",
            **self.universal_semantic_config,
        )
        
        user_info = BaseUserInfo(semantic_spec=semantic_spec)
        search_result = hetero_market.search_learnware(user_info)
        single_result = search_result.get_single_results()

        print(f"Search result1:")
        assert len(single_result) == 1, f"Exact semantic search failed!"
        for search_item in single_result:
            semantic_spec1 = search_item.learnware.get_specification().get_semantic_spec()
            print("Choose learnware:", search_item.learnware.id)
            assert semantic_spec1["Name"]["Values"] == semantic_spec["Name"]["Values"], f"Exact semantic search failed!"

        semantic_spec["Name"]["Values"] = "laernwaer"
        user_info = BaseUserInfo(semantic_spec=semantic_spec)
        search_result = hetero_market.search_learnware(user_info)
        single_result = search_result.get_single_results()

        print(f"Search result2:")
        assert len(single_result) == self.learnware_num, f"Fuzzy semantic search failed!"
        for search_item in single_result:
            print("Choose learnware:", search_item.learnware.id)

    def test_hetero_stat_search(self, learnware_num=5):
        hetero_market = self.test_train_market_model(learnware_num, delete=False)
        print("Total Item:", len(hetero_market))
        
        user_dim = 15

        with tempfile.TemporaryDirectory(prefix="learnware_test_hetero") as test_folder:
            for idx, zip_path in enumerate(self.zip_path_list):
                with zipfile.ZipFile(zip_path, "r") as zip_obj:
                    zip_obj.extractall(path=test_folder)

                user_spec = RKMETableSpecification()
                user_spec.load(os.path.join(test_folder, "stat_spec.json"))
                z = user_spec.get_z()
                z = z[:, :user_dim]
                device = user_spec.device
                z = torch.tensor(z, device=device)
                user_spec.z = z

                print(">> normal case test:")
                semantic_spec = generate_semantic_spec(
                    input_description={
                        "Dimension": user_dim,
                        "Description": {str(key): input_description_list[idx % 2]["Description"][str(key)] for key in range(user_dim)},
                    },
                    **self.universal_semantic_config,
                )
                user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})
                search_result = hetero_market.search_learnware(user_info)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()
                
                print(f"search result of user{idx}:")
                for single_item in single_result:
                    print(f"score: {single_item.score}, learnware_id: {single_item.learnware.id}")

                for multiple_item in multiple_result:
                    print(
                        f"mixture_score: {multiple_item.score}, mixture_learnware_ids: {[item.id for item in multiple_item.learnwares]}"
                    )

                # inproper key "Task" in semantic_spec, use homo search and print invalid semantic_spec
                print(">> test for key 'Task' has empty 'Values':")
                semantic_spec["Task"] = {"Values": ["Segmentation"], "Type": "Class"}
                user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})
                search_result = hetero_market.search_learnware(user_info)
                single_result = search_result.get_single_results()

                assert len(single_result) == 0, f"Statistical search failed!"

                # delete key "Task" in semantic_spec, use homo search and print WARNING INFO with "User doesn't provide correct task type"
                print(">> delele key 'Task' test:")
                semantic_spec.pop("Task")
                user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})
                search_result = hetero_market.search_learnware(user_info)
                single_result = search_result.get_single_results()

                assert len(single_result) == 0, f"Statistical search failed!"

                # modify semantic info with mismatch dim, use homo search and print "User data feature dimensions mismatch with semantic specification."
                print(">> mismatch dim test")
                semantic_spec = generate_semantic_spec(
                    input_description={
                        "Dimension": user_dim - 2,
                        "Description": {str(key): input_description_list[idx % 2]["Description"][str(key)] for key in range(user_dim)},
                    },
                    **self.universal_semantic_config,
                )
                user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})
                search_result = hetero_market.search_learnware(user_info)
                single_result = search_result.get_single_results()

                assert len(single_result) == 0, f"Statistical search failed!"

    def test_homo_stat_search(self, learnware_num=5):
        hetero_market = self.test_train_market_model(learnware_num, delete=False)
        print("Total Item:", len(hetero_market))
        
        with tempfile.TemporaryDirectory(prefix="learnware_test_hetero") as test_folder:
            for idx, zip_path in enumerate(self.zip_path_list):
                with zipfile.ZipFile(zip_path, "r") as zip_obj:
                    zip_obj.extractall(path=test_folder)

                user_spec = RKMETableSpecification()
                user_spec.load(os.path.join(test_folder, "stat_spec.json"))
                user_semantic = generate_semantic_spec(**self.universal_semantic_config)
                user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec})
                search_result = hetero_market.search_learnware(user_info)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                assert len(single_result) >= 1, f"Statistical search failed!"
                print(f"search result of user{idx}:")
                for single_item in single_result:
                    print(f"score: {single_item.score}, learnware_id: {single_item.learnware.id}")

                for multiple_item in multiple_result:
                    print(f"mixture_score: {multiple_item.score}\n")
                    mixture_id = " ".join([learnware.id for learnware in multiple_item.learnwares])
                    print(f"mixture_learnware: {mixture_id}\n")

    def test_model_reuse(self, learnware_num=5):
        # generate toy regression problem
        X, y = make_regression(n_samples=5000, n_informative=10, n_features=15, noise=0.1, random_state=0)

        # generate rkme
        user_spec = generate_rkme_table_spec(X=X, gamma=0.1, cuda_idx=0)

        # generate specification
        semantic_spec = generate_semantic_spec(input_description=user_description_list[0], **self.universal_semantic_config)
        user_info = BaseUserInfo(semantic_spec=semantic_spec, stat_info={"RKMETableSpecification": user_spec})

        # learnware market search
        hetero_market = self.test_train_market_model(learnware_num, delete=False)
        search_result = hetero_market.search_learnware(user_info)
        single_result = search_result.get_single_results()
        multiple_result = search_result.get_multiple_results()
        
        # print search results
        for single_item in single_result:
            print(f"score: {single_item.score}, learnware_id: {single_item.learnware.id}")

        for multiple_item in multiple_result:
            print(
                f"mixture_score: {multiple_item.score}, mixture_learnware_ids: {[item.id for item in multiple_item.learnwares]}"
            )

        # single model reuse
        hetero_learnware = HeteroMapAlignLearnware(single_result[0].learnware, mode="regression")
        hetero_learnware.align(user_spec, X[:100], y[:100])
        single_predict_y = hetero_learnware.predict(X)

        # multi model reuse
        hetero_learnware_list = []
        for learnware in multiple_result[0].learnwares:
            hetero_learnware = HeteroMapAlignLearnware(learnware, mode="regression")
            hetero_learnware.align(user_spec, X[:100], y[:100])
            hetero_learnware_list.append(hetero_learnware)

        # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = AveragingReuser(learnware_list=hetero_learnware_list, mode="mean")
        ensemble_predict_y = reuse_ensemble.predict(user_data=X)

        # Use ensemble pruning reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = EnsemblePruningReuser(learnware_list=hetero_learnware_list, mode="regression")
        reuse_ensemble.fit(X[:100], y[:100])
        ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=X)

        print("Single model RMSE by finetune:", mean_squared_error(y, single_predict_y, squared=False))
        print("Averaging Reuser RMSE:", mean_squared_error(y, ensemble_predict_y, squared=False))
        print("Ensemble Pruning Reuser RMSE:", mean_squared_error(y, ensemble_pruning_predict_y, squared=False))


def suite():
    _suite = unittest.TestSuite()
    #_suite.addTest(TestHeteroWorkflow("test_prepare_learnware_randomly"))
    #_suite.addTest(TestHeteroWorkflow("test_upload_delete_learnware"))
    #_suite.addTest(TestHeteroWorkflow("test_train_market_model"))
    _suite.addTest(TestHeteroWorkflow("test_search_semantics"))
    _suite.addTest(TestHeteroWorkflow("test_hetero_stat_search"))
    _suite.addTest(TestHeteroWorkflow("test_homo_stat_search"))
    _suite.addTest(TestHeteroWorkflow("test_model_reuse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
