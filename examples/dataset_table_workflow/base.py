import os
import time
import torch
import random
import requests
import tempfile
import traceback
import numpy as np
from queue import Empty
from tqdm import tqdm
from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market
from learnware.reuse.utils import fill_data_with_mean
from learnware.tests.benchmarks import LearnwareBenchmark
from torch.multiprocessing import Process, Queue, set_start_method

from config import *
from methods import *
from utils import process_single_aug

logger = get_module_logger("base_table", level="INFO")

try:
    set_start_method('spawn')
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy("file_system")


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
    def get_train_subsets(n_labeled_list, n_repeat_list, idx, train_x, train_y):
        np.random.seed(idx)
        random.seed(idx)
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
    
    @staticmethod
    def process_learnware_chunk(cuda_idx, method, test_info, loss_func, learnware_chunk, queue):
        torch.cuda.set_device(cuda_idx)
        for learnware in learnware_chunk:
            learnware_index = test_info['learnwares'].index(learnware)
            test_info['single_learnware'] = learnware
            scores = TableWorkflow._limited_data(method, test_info, loss_func)
            torch.cuda.empty_cache()
            queue.put((learnware_index, scores))
        
    def test_method(self, test_info, recorders, loss_func=loss_func_rmse):
        method_name_full = test_info["method_name"]
        method_name = method_name_full if method_name_full == "user_model" else "_".join(method_name_full.split("_")[1:])
        method = test_methods[method_name_full]
        user, idx = test_info["user"], test_info["idx"]
        recorder = recorders[method_name_full]
        
        save_root_path = os.path.join(self.curves_result_path, f"{user}/{user}_{idx}")
        os.makedirs(save_root_path, exist_ok=True)
        save_path = os.path.join(save_root_path, f"{method_name}.json")
        
        if method_name_full == "hetero_single_aug":
            if recorder.should_test_method(user, idx, save_path):
                # * single-process
                # bar = tqdm(total=len(test_info["learnwares"]), desc=f"Test {method_name}")
                # for learnware in test_info['learnwares']:
                #     test_info['single_learnware'] = learnware
                #     scores = self._limited_data(test_methods[method_name_full], test_info, loss_func)
                #     recorder.record(user, idx, scores)
                #     bar.update(1)  
                
                # * multi-process
                queue = Queue()
                processes = []
                bar = tqdm(total=len(test_info["learnwares"]), desc=f"Test {method_name}", unit="learnware")
                learnware_chunks = [test_info["learnwares"][i:len(test_info["learnwares"]):len(self.cuda_idx)] for i in self.cuda_idx]
                
                for cuda_idx, learnware_chunk in zip(self.cuda_idx, learnware_chunks):
                    p = Process(target=TableWorkflow.process_learnware_chunk, args=(cuda_idx, method, test_info, loss_func, learnware_chunk, queue))
                    processes.append(p)
                    p.start()
                
                all_results = []
                while any(p.is_alive() for p in processes) or not queue.empty():
                    try:
                        result = queue.get(timeout=0.1)
                        all_results.append(result)
                        bar.update(1)
                    except Empty:
                        time.sleep(0.1)
                        continue
                bar.close()

                for p in processes:
                    p.join()
                
                all_results.sort(key=lambda x: x[0])
                all_scores = [result[1] for result in all_results]
                recorder.record(user, all_scores)
                recorder.save(save_path)
                
            process_single_aug(user, idx, recorder.data[user][idx], recorders, save_root_path)
        else:
            if recorder.should_test_method(user, idx, save_path):
                scores = self._limited_data(method, test_info, loss_func)
                recorder.record(user, scores)
                recorder.save(save_path)
        
        logger.info(f"Method {method_name} on {user}_{idx} finished")