import os
import json
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pandas as pd


def get_data(data_root):
    X_train, y_train = fetch_20newsgroups(data_home=data_root, subset='train', return_X_y=True)
    X_test, y_test = fetch_20newsgroups(data_home=data_root, subset='test', return_X_y=True)

    return X_train, y_train, X_test, y_test