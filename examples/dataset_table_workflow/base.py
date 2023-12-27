import os
import time
import pandas
import random
import tempfile
import numpy as np
from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market
from learnware.tests.benchmarks import LearnwareBenchmark

from config import *
from methods import *
from utils import process_single_aug

logger = get_module_logger("base_table", level="INFO")


class TableWorkflow:
    def __init__(self, benchmark_config, name="easy", rebuild=False):
        self.root_path = os.path.abspath(os.path.join(__file__, ".."))
        self.result_path = os.path.join(self.root_path, "results")
        self.curves_result_path = os.path.join(self.root_path, "curves")
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.curves_result_path, exist_ok=True)
        self._prepare_market(benchmark_config, name, rebuild)
    
    @staticmethod
    def _limited_data(method, test_info, loss_func):
        all_scores = []
        for subset in test_info["train_subsets"]:
            subset_scores = []
            for sample in subset:
                x_train, y_train = sample["x_train"], sample["y_train"]
                model = method(x_train, y_train, test_info)
                subset_scores.append(loss_func(model.predict(test_info["test_x"]), test_info["test_y"]))
            all_scores.append(subset_scores)
        return all_scores
    
    @staticmethod
    def get_train_subsets(idx, train_x, train_y):
        np.random.seed(idx)
        random.seed(idx)
        train_subsets = []
        for n_label, repeated in zip(n_labeled_list, n_repeat_list):
            train_subsets.append([])
            if n_label > len(train_x):
                n_label = len(train_x)
            for _ in range(repeated):
                x_train, y_train = zip(*random.sample(list(zip(train_x, train_y)), k=n_label))
                train_subsets[-1].append({"x_train": np.array(x_train), "y_train": np.array(list(y_train))})
        return train_subsets
    
    def _prepare_market(self, benchmark_config, name, rebuild):
        client = LearnwareClient()
        self.benchmark = LearnwareBenchmark().get_benchmark(benchmark_config)
        self.market = instantiate_learnware_market(market_id=self.benchmark.name, name=name, rebuild=rebuild)
        self.user_semantic = client.get_semantic_specification(self.benchmark.learnware_ids[0])
        self.user_semantic["Name"]["Values"] = ""

        if len(self.market) == 0 or rebuild == True:
            for learnware_id in self.benchmark.learnware_ids:
                with tempfile.TemporaryDirectory(prefix="table_benchmark_") as tempdir:
                    zip_path = os.path.join(tempdir, f"{learnware_id}.zip")
                    for i in range(20):
                        try:
                            semantic_spec = client.get_semantic_specification(learnware_id)
                            client.download_learnware(learnware_id, zip_path)
                            self.market.add_learnware(zip_path, semantic_spec)
                            break
                        except:
                            time.sleep(1)
                            continue
    
    def test_method(self, test_info, recorders, loss_func=loss_func_rmse):
        method_name_full = test_info["method_name"]
        method_name = method_name_full if method_name_full == "user_model" else "_".join(method_name_full.split("_")[1:])
        user, idx = test_info["user"], test_info["idx"]
        recorder = recorders[method_name_full]
        
        save_root_path = os.path.join(self.curves_result_path, user, f"{user}_{idx}")
        os.makedirs(save_root_path, exist_ok=True)
        save_path = os.path.join(save_root_path, f"{method_name}.json")
        
        if method_name_full == "hetero_single_aug":
            if test_info["force"] or recorder.should_test_method(user, idx, save_path):
                for learnware in test_info["learnwares"]:
                    test_info["single_learnware"] = [learnware]
                    scores = self._limited_data(test_methods[method_name_full], test_info, loss_func)
                    recorder.record(user, scores)

                process_single_aug(user, idx, scores, recorders, save_root_path)
                recorder.save(save_path)
            else:
                process_single_aug(user, idx, recorder.data[user], recorders, save_root_path)
        else:
            if test_info["force"] or recorder.should_test_method(user, idx, save_path):
                scores = self._limited_data(test_methods[method_name_full], test_info, loss_func)
                recorder.record(user, scores)
                recorder.save(save_path)

        logger.info(f"Method {method_name} on {user}_{idx} finished")