from math import gamma
from tkinter import Y
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
import os, sys, gc, time, warnings, pickle, psutil, random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from .config import *


class AuxiliarySVR:
    def __init__(
        self, C, epsilon, gamma, adaptation_model=[], max_iter=30000, cache_size=10240, verbose=False, K1=None, K2=None
    ):
        self.gamma = gamma
        self.adaptation_model = adaptation_model
        self.model = SVR(
            C=C,
            epsilon=epsilon,
            kernel=self.auxiliary_rbf_kernel,
            max_iter=max_iter,
            cache_size=cache_size,
            verbose=verbose,
        )
        self.K1 = K1
        self.K2 = K2

    def auxiliary_rbf_kernel(self, X1, X2):
        if self.K1 is not None:
            if X1.shape[0] == X2.shape[0]:
                return self.K1[-X1.shape[0] :, -X2.shape[0] :]
            else:
                return self.K2[-X1.shape[0] :, -X2.shape[0] :]
        else:
            K = np.zeros((len(X1), len(X2)))

            for algo, idx in self.adaptation_model:
                Y1 = model_predict(algo, idx, X1).reshape(-1, 1)
                Y2 = model_predict(algo, idx, X2).reshape(-1, 1)
                K += Y1 @ Y2.T

            K += rbf_kernel(X1, X2, self.gamma)
            return K

    def fit(self, X, Y):
        self.gamma = 1 / X.shape[1]
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)


def measure_aux_algo(idx, test_sample, model):
    """
    model = ("lgb", 1)
    """
    store = store_list[idx]
    org_train_x, org_train_y, val_x, val_y = acquire_data(store, True)
    pred_y = model_predict(model[0], model[1], val_x[-test_sample:])
    return score(pred_y, val_y[-test_sample:])


# Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2.0 ** 30, 2)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


# Memory Reducer
def reduce_mem_usage(df, float16_flag=True, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if float16_flag and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how="left")
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def model_predict(algo, idx, test_x):
    store = store_list[idx]

    if algo == "lgb":
        model = lgb.Booster(model_file=os.path.join(model_dir, f"lgb_{store}.out"))
        return model.predict(test_x, num_iteration=model.best_iteration)
    elif algo == "ridge":
        model = joblib.load(os.path.join(model_dir, f"ridge_{store}.out"))
        return model.predict(test_x)
    elif algo == "svm":
        model = joblib.load(os.path.join(model_dir, f"svm_{store}.out"))
        return model.predict(test_x)


def get_weights(algo):
    weights = []

    if algo == "lgb":
        for store in store_list:
            model = lgb.Booster(model_file=os.path.join(model_dir, f"lgb_{store}.out"))
            weights.append(model.feature_importance())
    else:
        for store in store_list:
            model = joblib.load(os.path.join(model_dir, f"ridge_{store}.out"))
            weights.append(model.coef_)

    return np.array(weights)


def score(real_y, pred_y, sample_weight, multioutput):
    return mean_squared_error(real_y, pred_y, sample_weight=sample_weight, multioutput=multioutput, squared=False)


def acquire_data(store, fill_flag=False):
    TARGET = "sales"
    suffix = f"_fill" if fill_flag else ""
    train = pd.read_pickle(os.path.join(processed_data_dir, f"train_{store}{suffix}.pkl"))
    val = pd.read_pickle(os.path.join(processed_data_dir, f"val_{store}{suffix}.pkl"))

    train_y = train[TARGET]
    train_x = train.drop(columns=TARGET, axis=1)
    val_y = val[TARGET]
    val_x = val.drop(columns=TARGET, axis=1)

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    val_x = val_x.to_numpy()
    val_y = val_y.to_numpy()

    return train_x, train_y, val_x, val_y
