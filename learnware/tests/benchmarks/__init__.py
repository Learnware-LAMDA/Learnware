import os
import pickle
import atexit
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple

from .config import OnlineBenchmark, online_benchmarks
from ..data import GetData
@dataclass
class Benchmark:
    learnware_ids: List[str]
    user_num: int
    unlabeled_feature_paths: List[str]
    unlabeled_groudtruths_paths: List[str]
    labeled_feature_paths: Optional[List[str]] = None
    labeled_label_paths: Optional[List[str]] = None
    extra_info_path: Optional[str] = None
    
    # TODO: add more method for benchmark
    
    def get_unlabeled_data(self, user_ids: Union[str, List[str]]):
        if isinstance(user_ids, str):
            user_ids = [user_ids]
        
        ret = []
        for user_id in user_ids:
            with open(self.unlabeled_feature_paths[user_id], "rb") as fin:
                unlabeled_feature = pickle.load(fin)
            
            with open(self.unlabeled_groudtruths_paths[user_id], "rb") as fin:
                unlabeled_groudtruth = pickle.load(fin)
        
            ret.append((unlabeled_feature, unlabeled_groudtruth))

        return ret
    
    def get_labeled_data(self, user_ids):
        if self.labeled_feature_paths is None or self.labeled_label_paths is None:
            return None
        
        if isinstance(user_ids, str):
            user_ids = [user_ids]
        
        ret = []
        for user_id in user_ids:
            with open(self.labeled_feature_paths[user_id], "rb") as fin:
                labeled_feature = pickle.load(fin)
            
            with open(self.labeled_label_paths[user_id], "rb") as fin:
                labeled_groudtruth = pickle.load(fin)
        
            ret.append((labeled_feature, labeled_groudtruth))

        return ret

            
class LearnwareBenchmark:
    
    def __init__(self):
        self.online_benchmarks = online_benchmarks
        self.tempdir_list = []
        atexit.register(self.cleanup)
        
    def list_benchmarks(self):
        return list(self.online_benchmarks.keys())
        
    def get_benchmark(self, online_benchmark: Union[str, OnlineBenchmark]):
        if isinstance(online_benchmark, str):
            online_benchmark = self.online_benchmarks[online_benchmark]
        
        self.tempdir_list.append(tempfile.TemporaryDirectory(prefix="learnware_benchmark"))
        save_folder = self.tempdir_list[-1].name
    
        unlabeled_data_localpath = os.path.join(save_folder, "unlabeled_data.zip")
        GetData().download_file(online_benchmark.unlabeled_data_path, unlabeled_data_localpath)
        
        unlabeled_feature_paths = []
        unlabeled_groudtruth_paths = []
        
        with zipfile.ZipFile(unlabeled_data_localpath, "r") as z_file:
            unlabeled_data_dirpath = os.path.join(save_folder, "unlabeled_data")
            z_file.extractall(unlabeled_data_dirpath)
            for user_id in range(online_benchmark.user_num):
                user_feature_filepath = os.path.isfile(os.path.join(unlabeled_data_dirpath, f"user{user_id}_feature.pkl"))
                user_groudtruth_filepath = os.path.isfile(os.path.join(unlabeled_data_dirpath, f"user{user_id}_groudtruth.pkl"))
                assert os.path.isfile(user_feature_filepath), f"user {user_id} unlabeled feature is not valid!"
                assert os.path.isfile(user_groudtruth_filepath), f"user {user_id} unlabeled groudtruth is not valid!"
                unlabeled_feature_paths.append(user_feature_filepath)
                unlabeled_groudtruth_paths.append(user_groudtruth_filepath)

        labeled_feature_paths = None
        labeled_label_paths = None
        if online_benchmark.labeled_data_path is not None:
            labeled_data_localpath = os.path.join(save_folder, "labeled_data.zip")
            GetData().download_file(online_benchmark.labeled_data_path, labeled_data_localpath)
            
            labeled_feature_paths = []
            labeled_label_paths = []

            with zipfile.ZipFile(labeled_data_localpath, "r") as z_file:
                labeled_data_dirpath = os.path.join(save_folder, "labeled_data")
                z_file.extractall(labeled_data_dirpath)
                for user_id in range(online_benchmark.user_num):
                    user_feature_filepath = os.path.isfile(os.path.join(labeled_data_dirpath, f"user{user_id}_feature.pkl"))
                    user_groudtruth_filepath = os.path.isfile(os.path.join(labeled_data_dirpath, f"user{user_id}_label.pkl"))
                    assert os.path.isfile(user_feature_filepath), f"user {user_id} labeled feature is not valid!"
                    assert os.path.isfile(user_groudtruth_filepath), f"user {user_id} labeled label is not valid!"
                    labeled_feature_paths.append(user_feature_filepath)
                    labeled_label_paths.append(user_groudtruth_filepath)
                    
        extra_zip_localpath = None
        if online_benchmark.extra_info_path is not None:
            extra_zip_localpath = os.path.join(save_folder, os.path.basename(online_benchmark.extra_info_path))
            GetData().download_file(online_benchmark.extra_info_path, extra_zip_localpath)
            
        return Benchmark(
            learnware_ids=online_benchmark.learnware_ids,
            user_num=online_benchmark.user_num,
            unlabeled_feature_paths=unlabeled_feature_paths,
            unlabeled_groudtruths_paths=unlabeled_groudtruth_paths,
            labeled_feature_paths=labeled_feature_paths,
            labeled_label_paths=labeled_label_paths,
            extra_info_path=extra_zip_localpath,
        )
    
    def cleanup(self):
        for tempdir in self.tempdir_list:
            tempdir.cleanup()