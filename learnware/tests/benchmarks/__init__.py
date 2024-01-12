import os
import pickle
import tempfile
import zipfile
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .config import BenchmarkConfig, benchmark_configs
from ..data import GetData
from ...config import C


@dataclass
class Benchmark:
    name: str
    user_num: int
    learnware_ids: List[str]
    test_X_paths: List[str]
    test_y_paths: List[str]
    train_X_paths: Optional[List[str]] = None
    train_y_paths: Optional[List[str]] = None
    extra_info_path: Optional[str] = None

    def get_test_data(
        self, user_ids: Union[int, List[int]]
    ) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        raw_user_ids = user_ids
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        ret = []
        for user_id in user_ids:
            with open(self.test_X_paths[user_id], "rb") as fin:
                test_X = pickle.load(fin)

            with open(self.test_y_paths[user_id], "rb") as fin:
                test_y = pickle.load(fin)

            ret.append((test_X, test_y))

        if isinstance(raw_user_ids, int):
            return ret[0]
        else:
            return ret

    def get_train_data(
        self, user_ids: Union[int, List[int]]
    ) -> Optional[Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]]:
        if self.train_X_paths is None or self.train_y_paths is None:
            return None

        raw_user_ids = user_ids
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        ret = []
        for user_id in user_ids:
            with open(self.train_X_paths[user_id], "rb") as fin:
                train_X = pickle.load(fin)

            with open(self.train_y_paths[user_id], "rb") as fin:
                train_y = pickle.load(fin)

            ret.append((train_X, train_y))

        if isinstance(raw_user_ids, int):
            return ret[0]
        else:
            return ret


class LearnwareBenchmark:
    def __init__(self):
        self.benchmark_configs = benchmark_configs

    def list_benchmarks(self):
        return list(self.benchmark_configs.keys())

    def _check_cache_data_valid(self, benchmark_config: BenchmarkConfig, data_type: str) -> bool:
        """Check if the cache data is valid

        Parameters
        ----------
        benchmark_config : BenchmarkConfig
            benchmark config
        data_type : str
            "test" for test data or "train" for train data

        Returns
        -------
        bool
            A flag indicating if the cache data is valid
        """
        cache_folder = os.path.join(C.cache_path, benchmark_config.name, f"{data_type}_data")
        if os.path.exists(cache_folder):
            for user_id in range(benchmark_config.user_num):
                X_path = os.path.join(cache_folder, f"user{user_id}_X.pkl")
                y_path = os.path.join(cache_folder, f"user{user_id}_X.pkl")
                if not os.path.isfile(X_path) or not os.path.isfile(y_path):
                    return False
            return True
        else:
            return False

    def _download_data(self, download_path: str, save_path: str) -> None:
        """Download data from backend

        Parameters
        ----------
        download_path : str
            data path for download in backend
        save_path : str
            local cache path for saving data
        """
        with tempfile.TemporaryDirectory(prefix="learnware_benchmark_") as tempdir:
            test_data_zippath = os.path.join(tempdir, "benchmark_data.zip")
            GetData().download_file(download_path, test_data_zippath)

            os.makedirs(save_path, exist_ok=True)
            with zipfile.ZipFile(test_data_zippath, "r") as z_file:
                z_file.extractall(save_path)

    def _load_cache_data(self, benchmark_config: BenchmarkConfig, data_type: str) -> Tuple[List[str], List[str]]:
        """Load data from local cache path

        Parameters
        ----------
        benchmark_config : BenchmarkConfig
            benchmark config
        data_type : str
            "test" for test data or "train" for train data
        """
        cache_folder = os.path.join(C.cache_path, benchmark_config.name, f"{data_type}_data")
        if not self._check_cache_data_valid(benchmark_config, data_type):
            download_path = getattr(benchmark_config, f"{data_type}_data_path")
            self._download_data(download_path, cache_folder)

        X_paths, y_paths = [], []
        for user_id in range(benchmark_config.user_num):
            user_X_path = os.path.join(cache_folder, f"user{user_id}_X.pkl")
            user_y_path = os.path.join(cache_folder, f"user{user_id}_y.pkl")
            assert os.path.isfile(user_X_path), f"user {user_id} {data_type}_X is not valid!"
            assert os.path.isfile(user_y_path), f"user {user_id} {data_type}_y is not valid!"
            X_paths.append(user_X_path)
            y_paths.append(user_y_path)

        return X_paths, y_paths

    def get_benchmark(self, benchmark_config: Union[str, BenchmarkConfig]) -> Benchmark:
        if isinstance(benchmark_config, str):
            benchmark_config = self.benchmark_configs[benchmark_config]

        if not isinstance(benchmark_config, BenchmarkConfig):
            raise ValueError(
                "benchmark_config must be a BenchmarkConfig object or a string in benchmark_configs.keys()!"
            )

        # Load test data
        test_X_paths, test_y_paths = self._load_cache_data(benchmark_config, "test")

        # Load train data
        train_X_paths, train_y_paths = None, None
        if benchmark_config.train_data_path is not None:
            train_X_paths, train_y_paths = self._load_cache_data(benchmark_config, "train")

        # Load extra info
        extra_info_path = None
        if benchmark_config.extra_info_path is not None:
            extra_info_path = os.path.join(C.cache_path, benchmark_config.name, "extra_info")
            if not os.path.exists(extra_info_path):
                self._download_data(benchmark_config.extra_info_path, extra_info_path)

        return Benchmark(
            name=benchmark_config.name,
            user_num=benchmark_config.user_num,
            learnware_ids=benchmark_config.learnware_ids,
            test_X_paths=test_X_paths,
            test_y_paths=test_y_paths,
            train_X_paths=train_X_paths,
            train_y_paths=train_y_paths,
            extra_info_path=extra_info_path,
        )
