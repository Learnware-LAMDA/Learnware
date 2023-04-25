from cgi import test
import os
import joblib
import lightgbm as lgb


from .config import store_list, model_dir
from .utils import acquire_data, get_weights, model_predict, score, measure_aux_algo
from .generate_data import regenerate_data
from .train import retrain_models, grid_training_sample, train_adaptation_grid


class DataLoader:
    def __init__(self):
        self.algo = "ridge"

    def set_algo(self, algo):
        self.algo = algo

    def get_algo_list(self):
        return ["lgb", "ridge"]

    def get_idx_list(self):
        return list(range(len(store_list)))

    def get_idx_data(self, idx):
        store = store_list[idx]
        # fill_flag = self.algo == "ridge"
        fill_flag = True
        return acquire_data(store, fill_flag)

    def get_weights(self):
        return get_weights(self.algo)

    def get_model_path(self, idx):
        return os.path.join(model_dir, "{}_{}.out".format(self.algo, store_list[idx]))

    def predict(self, idx, test_x):
        store = store_list[idx]

        if os.path.exists(os.path.join(model_dir, f"{self.algo}_{store}.out")):
            return model_predict(self.algo, idx, test_x)
        else:
            self.retrain_models()
            return model_predict(self.algo, idx, test_x)

    def score(self, real_y, pred_y, sample_weight=None, multioutput="raw_values"):
        return score(real_y, pred_y, sample_weight, multioutput)

    def regenerate_data(self):
        regenerate_data()

    def retrain_models(self):
        retrain_models(self.algo)

    def grid_training_sample(self, user_list=list(range(10))):
        grid_training_sample(self.algo, user_list)

    def train_adaptation_grid(
        self, max_sample, test_sample, user_list=list(range(10)), adaptation_model=[], residual=False
    ):
        train_adaptation_grid(self.algo, max_sample, test_sample, user_list, adaptation_model, residual)

    def measure_aux_algo(self, idx, test_sample, model):
        return measure_aux_algo(idx, test_sample, model)
