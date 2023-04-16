import joblib
import os
from sklearn.metrics import mean_squared_error


from .pfs_cross_transfer import *
from .split_data import feature_engineering


class Dataloader:
    def __init__(self):
        self.algo = "ridge"

    def regenerate_data(self):
        feature_engineering()

    def set_algo(self, algo):
        self.algo = algo

    def get_algo_list(self):
        return ["lgb", "ridge"]

    def get_idx_list(self):
        return [i for i in range(53)]

    def get_idx_data(self, idx):
        shop_ids = [i for i in range(60) if i not in [0, 1, 40]]
        shop_ids = [i for i in shop_ids if i not in [8, 11, 23, 36]]

        fpath = os.path.join(pfs_split_dir, "Shop{:0>2d}-train.csv".format(shop_ids[idx]))
        train_xs, train_ys, _, _ = load_pfs_data(fpath)
        fpath = os.path.join(pfs_split_dir, "Shop{:0>2d}-val.csv".format(shop_ids[idx]))
        test_xs, test_ys, _, _ = load_pfs_data(fpath)
        return train_xs, train_ys, test_xs, test_ys

    def get_model_path(self, idx):
        shop_ids = [i for i in range(60) if i not in [0, 1, 40]]
        shop_ids = [i for i in shop_ids if i not in [8, 11, 23, 36]]
        return os.path.join(model_dir, "{}_Shop{:0>2d}.out".format(self.algo, shop_ids[idx]))
    
    def retrain_models(self):
        algo = self.algo
        errs = get_errors(algo=algo)

        fpath = os.path.join(pfs_res_dir, "PFS_{}_errs.txt".format(algo))
        np.savetxt(fpath, errs.T)

        plot_heatmap(errs.T, algo)
        weights = np.loadtxt(os.path.join(pfs_res_dir, "PFS_{}_weights.txt".format(algo)))
        plot_performance(errs.T, weights, algo)

    def retrain_split_models(self):
        fpath = os.path.join(pfs_res_dir, "PFS_{}_split_errs_user.txt".format(self.algo))
        if os.path.exists(fpath):
            return np.loadtxt(fpath)
        algo = self.algo
        errs = get_split_errs(algo=algo)
        fpath = os.path.join(pfs_res_dir, "PFS_{}_split_errs_user.txt".format(algo))
        np.savetxt(fpath, errs)
        return errs

    def get_errs(self):
        return np.loadtxt(os.path.join(pfs_res_dir, "PFS_{}_errs.txt".format(self.algo)))

    def get_weights(self):
        return np.loadtxt(os.path.join(pfs_res_dir, "PFS_{}_weights.txt".format(self.algo)))

    def predict(self, idx, test_x):
        shop_ids = [i for i in range(60) if i not in [0, 1, 40]]
        shop_ids = [i for i in shop_ids if i not in [8, 11, 23, 36]]

        model = joblib.load(os.path.join(model_dir, "{}_Shop{:0>2d}.out".format(self.algo, shop_ids[idx])))
        # test_x = (test_x - test_x.min(0)) / (test_x.max(0) - test_x.min(0) + 0.0001)
        return model.predict(test_x)

    def score(self, real_y, pred_y, sample_weight=None):
        return mean_squared_error(real_y, pred_y, sample_weight=sample_weight, squared=False)