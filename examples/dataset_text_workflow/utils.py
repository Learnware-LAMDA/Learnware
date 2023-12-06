import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, Booster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class TextDataLoader:
    def __init__(self, data_root, train: bool = True):
        self.data_root = data_root
        self.train = train

    def get_idx_data(self, idx=0):
        if self.train:
            X_path = os.path.join(self.data_root, "uploader", "uploader_%d_X.pkl" % (idx))
            y_path = os.path.join(self.data_root, "uploader", "uploader_%d_y.pkl" % (idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            with open(X_path, "rb") as f:
                X = pickle.load(f)
            with open(y_path, "rb") as f:
                y = pickle.load(f)
        else:
            X_path = os.path.join(self.data_root, "user", "user_%d_X.pkl" % (idx))
            y_path = os.path.join(self.data_root, "user", "user_%d_y.pkl" % (idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            with open(X_path, "rb") as f:
                X = pickle.load(f)
            with open(y_path, "rb") as f:
                y = pickle.load(f)
        return X, y


def generate_uploader(data_x: pd.Series, data_y: pd.Series, n_uploaders=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)

    types = data_x["discourse_type"].unique()

    for i in range(n_uploaders):
        indices = data_x["discourse_type"] == types[i]
        selected_X = (types[i] + ' ' + data_x[indices]["discourse_text"]).to_list()
        selected_y = data_y[indices].to_list()

        X_save_dir = os.path.join(data_save_root, "uploader_%d_X.pkl" % (i))
        y_save_dir = os.path.join(data_save_root, "uploader_%d_y.pkl" % (i))
        with open(X_save_dir, "wb") as f:
            pickle.dump(selected_X, f)
        with open(y_save_dir, "wb") as f:
            pickle.dump(selected_y, f)

        print("Saving to %s" % (X_save_dir))


def generate_user(data_x, data_y, n_users=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)

    types = data_x["discourse_type"].unique()

    for i in range(n_users):
        indices = data_x["discourse_type"] == types[i]
        selected_X = (types[i] + ' ' + data_x[indices]["discourse_text"]).to_list()
        selected_y = data_y[indices].to_list()

        X_save_dir = os.path.join(data_save_root, "user_%d_X.pkl" % (i))
        y_save_dir = os.path.join(data_save_root, "user_%d_y.pkl" % (i))
        with open(X_save_dir, "wb") as f:
            pickle.dump(selected_X, f)
        with open(y_save_dir, "wb") as f:
            pickle.dump(selected_y, f)

        print("Saving to %s" % (X_save_dir))


from param_search import grid_search


# Train Uploaders' models
def train(X, y, out_classes):
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    X_trian_tfidf = vectorizer.transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    model_path = "models/model.pkl"
    best_params = grid_search(X_trian_tfidf, X_valid_tfidf, y_train, y_valid, out_classes, True, model_path)

    # lgbm = Booster(model_file="models/model.txt")
    # with open(model_path, "rb") as f:
    #     lgbm = pickle.load(f)

    param = {
        "learning_rate": 0.1,
        "importance_type": "gain",
        "objective": "multiclass",
        "num_class": out_classes,
        "n_estimators": 1000,
        'max_bin': 512,
        "verbose": -1,
        **best_params}
    lgbm = LGBMClassifier(**param)
    lgbm.fit(X_tfidf, y)

    return vectorizer, lgbm


def eval_prediction(pred_y, target_y):
    if not isinstance(pred_y, np.ndarray):
        pred_y = pred_y.detach().cpu().numpy()
    if len(pred_y.shape) == 1:
        predicted = np.array(pred_y)
    else:
        predicted = np.argmax(pred_y, 1)
    annos = np.array(target_y)

    total = predicted.shape[0]
    correct = (predicted == annos).sum().item()

    return correct / total
