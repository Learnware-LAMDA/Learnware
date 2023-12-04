import unittest
import os
import logging
import tempfile
import pickle
import zipfile
import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import learnware
learnware.init(logging_level=logging.WARNING)

from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.specification import RKMETableSpecification, generate_rkme_table_spec, generate_semantic_spec
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser, FeatureAugmentReuser
from learnware.tests.templates import LearnwareTemplate, PickleModelTemplate, StatSpecTemplate

curr_root = os.path.dirname(os.path.abspath(__file__))

class TestWorkflow(unittest.TestCase):
    
    universal_semantic_config = {
        "data_type": "Table",
        "task_type": "Classification",
        "library_type": "Scikit-learn",
        "scenarios": "Education",
        "license": "MIT",
    }
    
    @classmethod
    def setUpClass(cls):
       pass
    
    def _init_learnware_market(self):
        """initialize learnware market"""
        easy_market = instantiate_learnware_market(market_id="sklearn_digits_easy", name="easy", rebuild=True)
        return easy_market

    def test_prepare_learnware_randomly(self, learnware_num=5):
        self.zip_path_list = []
        X, y = load_digits(return_X_y=True)

        for i in range(learnware_num):
            learnware_pool_dirpath = os.path.join(curr_root, "learnware_pool")
            os.makedirs(learnware_pool_dirpath, exist_ok=True)
            learnware_zippath = os.path.join(learnware_pool_dirpath, "svm_%d.zip" % (i))
            
            print("Preparing Learnware: %d" % (i))
            data_X, _, data_y, _ = train_test_split(X, y, test_size=0.3, shuffle=True)
            clf = svm.SVC(kernel="linear", probability=True)
            clf.fit(data_X, data_y)
            pickle_filepath = os.path.join(learnware_pool_dirpath, "model.pkl")
            with open(pickle_filepath, "wb") as fout:
                pickle.dump(clf, fout)

            spec = generate_rkme_table_spec(X=data_X, gamma=0.1, cuda_idx=0)
            spec_filepath = os.path.join(learnware_pool_dirpath, "stat_spec.json")
            spec.save(spec_filepath)
            
            LearnwareTemplate.generate_learnware_zipfile(
                learnware_zippath=learnware_zippath,
                model_template=PickleModelTemplate(pickle_filepath=pickle_filepath, model_kwargs={"input_shape":(64,), "output_shape": (10,), "predict_method": "predict_proba"}),
                stat_spec_template=StatSpecTemplate(filepath=spec_filepath, type="RKMETableSpecification")
            )
           
            self.zip_path_list.append(learnware_zippath)

    def test_upload_delete_learnware(self, learnware_num=5, delete=True):
        easy_market = self._init_learnware_market()
        self.test_prepare_learnware_randomly(learnware_num)
        self.learnware_num = learnware_num

        print("Total Item:", len(easy_market))
        assert len(easy_market) == 0, f"The market should be empty!"

        for idx, zip_path in enumerate(self.zip_path_list):
            semantic_spec = generate_semantic_spec(
                name=f"learnware_{idx}",
                description=f"test_learnware_number_{idx}",
                input_description={
                    "Dimension": 64,
                    "Description": {
                        f"{i}": f"The value in the grid {i // 8}{i % 8} of the image of hand-written digit."
                        for i in range(64)
                    },
                },
                output_description={
                    "Dimension": 10,
                    "Description": {f"{i}": "The probability for each digit for 0 to 9." for i in range(10)},
                },
                **self.universal_semantic_config
            )
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
        
        with tempfile.TemporaryDirectory(prefix="learnware_test_workflow") as test_folder:
            with zipfile.ZipFile(self.zip_path_list[0], "r") as zip_obj:
                zip_obj.extractall(path=test_folder)

            semantic_spec = generate_semantic_spec(
                name=f"learnware_{learnware_num - 1}",
                description=f"test_learnware_number_{learnware_num - 1}",
                **self.universal_semantic_config,
            )
            
            user_info = BaseUserInfo(semantic_spec=semantic_spec)
            search_result = easy_market.search_learnware(user_info)
            single_result = search_result.get_single_results()

            print(f"Search result:")
            for search_item in single_result:
                print("Choose learnware:",search_item.learnware.id)
      
    def test_stat_search(self, learnware_num=5):
        easy_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(easy_market))

        with tempfile.TemporaryDirectory(prefix="learnware_test_workflow") as test_folder:
            for idx, zip_path in enumerate(self.zip_path_list):
                with zipfile.ZipFile(zip_path, "r") as zip_obj:
                    zip_obj.extractall(path=test_folder)

                user_spec = RKMETableSpecification()
                user_spec.load(os.path.join(test_folder, "stat_spec.json"))
                user_semantic = generate_semantic_spec(**self.universal_semantic_config)
                user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_spec})
                search_results = easy_market.search_learnware(user_info)

                single_result = search_results.get_single_results()
                multiple_result = search_results.get_multiple_results()

                assert len(single_result) >= 1, f"Statistical search failed!"
                print(f"search result of user{idx}:")
                for search_item in single_result:
                    print(f"score: {search_item.score}, learnware_id: {search_item.learnware.id}")

                for mixture_item in multiple_result:
                    print(f"mixture_score: {mixture_item.score}\n")
                    mixture_id = " ".join([learnware.id for learnware in mixture_item.learnwares])
                    print(f"mixture_learnware: {mixture_id}\n")

    def test_learnware_reuse(self, learnware_num=5):
        easy_market = self.test_upload_delete_learnware(learnware_num, delete=False)
        print("Total Item:", len(easy_market))

        X, y = load_digits(return_X_y=True)
        train_X, data_X, train_y, data_y = train_test_split(X, y, test_size=0.3, shuffle=True)

        stat_spec = generate_rkme_table_spec(X=data_X, gamma=0.1, cuda_idx=0)
        user_semantic = generate_semantic_spec(**self.universal_semantic_config)
        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": stat_spec})

        search_results = easy_market.search_learnware(user_info)
        multiple_result = search_results.get_multiple_results()
        mixture_item = multiple_result[0]
        # Based on user information, the learnware market returns a list of learnwares (learnware_list)
        # Use jobselector reuser to reuse the searched learnwares to make prediction
        reuse_job_selector = JobSelectorReuser(learnware_list=mixture_item.learnwares)
        job_selector_predict_y = reuse_job_selector.predict(user_data=data_X)

        # Use averaging ensemble reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = AveragingReuser(learnware_list=mixture_item.learnwares, mode="vote_by_prob")
        ensemble_predict_y = reuse_ensemble.predict(user_data=data_X)

        # Use ensemble pruning reuser to reuse the searched learnwares to make prediction
        reuse_ensemble = EnsemblePruningReuser(learnware_list=mixture_item.learnwares, mode="classification")
        reuse_ensemble.fit(train_X[-200:], train_y[-200:])
        ensemble_pruning_predict_y = reuse_ensemble.predict(user_data=data_X)

        # Use feature augment reuser to reuse the searched learnwares to make prediction
        reuse_feature_augment = FeatureAugmentReuser(learnware_list=mixture_item.learnwares, mode="classification")
        reuse_feature_augment.fit(train_X[-200:], train_y[-200:])
        feature_augment_predict_y = reuse_feature_augment.predict(user_data=data_X)

        print("Job Selector Acc:", np.sum(np.argmax(job_selector_predict_y, axis=1) == data_y) / len(data_y))
        print("Averaging Reuser Acc:", np.sum(np.argmax(ensemble_predict_y, axis=1) == data_y) / len(data_y))
        print("Ensemble Pruning Reuser Acc:", np.sum(ensemble_pruning_predict_y == data_y) / len(data_y))
        print("Feature Augment Reuser Acc:", np.sum(feature_augment_predict_y == data_y) / len(data_y))


def suite():
    _suite = unittest.TestSuite()
    # _suite.addTest(TestWorkflow("test_prepare_learnware_randomly"))
    # _suite.addTest(TestWorkflow("test_upload_delete_learnware"))
    _suite.addTest(TestWorkflow("test_search_semantics"))
    _suite.addTest(TestWorkflow("test_stat_search"))
    _suite.addTest(TestWorkflow("test_learnware_reuse"))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
