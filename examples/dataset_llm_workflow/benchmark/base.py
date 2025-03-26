from .config import LEARNWARE_MATH, LEARNWARE_MED, USER_MED, USER_MATH, LEARNWARE_FIN, USER_FIN, LEARNWARE_MED_IDS, LEARNWARE_MATH_IDS, LEARNWARE_FIN_IDS
from .utils import prepare_train_data, prepare_test_data
from datasets import Dataset
from typing import List, Tuple

class Benchmark:
    def __init__(self, name: str):
        self.name = name
        self.set_datasets(name)
    
    def get_benchmark_name(self):
        return self.name

    def set_datasets(self, name: str):
        if name == "medical":
            self.learnware_dict = LEARNWARE_MED
            self.learnware_ids = LEARNWARE_MED_IDS
            self.user_dict = USER_MED
        elif name == "math":
            self.learnware_dict = LEARNWARE_MATH
            self.learnware_ids = LEARNWARE_MATH_IDS
            self.user_dict = USER_MATH
        elif name == "finance":
            self.learnware_dict = LEARNWARE_FIN
            self.learnware_ids = LEARNWARE_FIN_IDS
            self.user_dict = USER_FIN
        else:
            raise NotImplementedError("other benchmarks are not implemented")
    
    def get_learnware_ids(self) -> List[str]:
        return self.learnware_ids
    
    def get_learnware_data(self, dataset_name) -> List[str]:
        train_dataset, val_dataset = prepare_train_data(self.learnware_dict[dataset_name])
        train_data, val_data = train_dataset["text"], val_dataset["text"]
        return train_data, val_data
    
    def get_learnware_dataset(self, dataset_name) -> Tuple[Dataset, Dataset]:
        train_dataset, val_dataset = prepare_train_data(self.learnware_dict[dataset_name])
        return train_dataset, val_dataset
    
    def get_user_data(self, dataset_name) -> List[str]:
        test_dataset = prepare_test_data(self.user_dict[dataset_name])
        test_data = test_dataset["text"]
        return test_data
    
    def get_user_dataset(self, dataset_name) -> Dataset:
        test_dataset = prepare_test_data(self.user_dict[dataset_name])
        return test_dataset
    
    def get_learnwares(self):
        return self.learnware_dict
    
    def get_users(self):
        return self.user_dict
    
    def get_learnware_names(self) -> List[str]:
        return list(self.learnware_dict.keys())
    
    def get_user_names(self) -> List[str]:
        return list(self.user_dict.keys())
