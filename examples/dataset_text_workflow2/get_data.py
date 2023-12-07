import os
import json
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def get_data(data_root):
    dataset_train = fetch_20newsgroups(data_home=data_root, subset='train')
    target_names = dataset_train["target_names"]

    X_train = np.array(dataset_train["data"])
    y_train = pd.Categorical.from_codes(dataset_train["target"], categories=target_names)

    # y_train = [target_names[label] for label in dataset_train["target"]]

    X_test, y_test = fetch_20newsgroups(data_home=data_root, subset='test', return_X_y=True)
    X_test = np.array(X_test)
    y_test = pd.Categorical.from_codes(y_test, categories=target_names)

    return X_train, y_train, X_test, y_test