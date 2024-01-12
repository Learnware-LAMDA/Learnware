import os
import time
import torch
import random
import requests
import tempfile
import traceback
import numpy as np
from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market
from learnware.reuse.utils import fill_data_with_mean
from learnware.tests.benchmarks import LearnwareBenchmark

from config import *
from methods import *

logger = get_module_logger("base_table", level="INFO")


class TableWorkflow:
    def __init__(self, benchmark_config, name="easy", rebuild=False, retrain=False):
        self.root_path = os.path.abspath(os.path.join(__file__, ".."))
        self.result_path = os.path.join(self.root_path, "results")
        self.curves_result_path = os.path.join(self.result_path, "curves")
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.curves_result_path, exist_ok=True)
        self._prepare_market(benchmark_config, name, rebuild, retrain)
        
        self.cuda_idx = list(range(torch.cuda.device_count()))
    
    @staticmethod
    def _limited_data(method, test_info, loss_func):
        def subset_generator():
            for subset in test_info["train_subsets"]:
                yield subset
                
        all_scores = []
        for subset in subset_generator():
            subset_scores = []
            for sample in subset:
                x_train, y_train = sample["x_train"], sample["y_train"]
                model = method(x_train, y_train, test_info)
                subset_scores.append(loss_func(model.predict(test_info["test_x"]), test_info["test_y"]))
            all_scores.append(subset_scores)
        return all_scores
    
    @staticmethod
    def get_train_subsets(n_labeled_list, n_repeat_list, train_x, train_y):
        np.random.seed(1)
        random.seed(1)
        train_x = fill_data_with_mean(train_x)
        train_subsets = []
        for n_label, repeated in zip(n_labeled_list, n_repeat_list):
            train_subsets.append([])
            if n_label > len(train_x):
                n_label = len(train_x)
            for _ in range(repeated):
                subset_idxs = np.random.choice(len(train_x), n_label, replace=False)
                train_subsets[-1].append({"x_train": np.array(train_x[subset_idxs]), "y_train": np.array(train_y[subset_idxs])})
        return train_subsets
    
    def _prepare_market(self, benchmark_config, name, rebuild, retrain):
        client = LearnwareClient()
        self.benchmark = LearnwareBenchmark().get_benchmark(benchmark_config)
        self.market = instantiate_learnware_market(
            market_id=self.benchmark.name,
            name=name,
            rebuild=rebuild,
            organizer_kwargs={
                "auto_update": True,
                "auto_update_limit": len(self.benchmark.learnware_ids),
                **market_mapping_params
            } if retrain else None
        )
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
                        except (requests.exceptions.RequestException, IOError, Exception) as e:
                            logger.info(f"An error occurred when downloading {learnware_id}: {e}\n{traceback.format_exc()}, retrying...")
                            time.sleep(1)
                            continue
        
    def test_method(self, test_info, recorders, loss_func=loss_func_rmse):
        method_name_full = test_info["method_name"]
        method_name = method_name_full if method_name_full == "user_model" else "_".join(method_name_full.split("_")[1:])
        method = test_methods[method_name_full]
        user, idx = test_info["user"], test_info["idx"]
        recorder = recorders[method_name_full]
        
        save_root_path = os.path.join(self.curves_result_path, f"{user}/{user}_{idx}")
        os.makedirs(save_root_path, exist_ok=True)
        save_path = os.path.join(save_root_path, f"{method_name}.json")
        
        if recorder.should_test_method(user, idx, save_path):
            scores = self._limited_data(method, test_info, loss_func)
            recorder.record(user, scores)
            recorder.save(save_path)
        
        logger.info(f"Method {method_name} on {user}_{idx} finished")