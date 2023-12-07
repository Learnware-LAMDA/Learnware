import os
import pickle
import random
from itertools import combinations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, Booster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

super_classes = ["comp", "rec", "sci", "talk", "misc"]
super_classes_select2 = list(combinations(super_classes, 2))
super_classes_select3 = list(combinations(super_classes, 3))


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


def generate_uploader(data_x, data_y, n_uploaders=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)
    n = len(data_x)

    for i, labels in enumerate(super_classes_select3[:n_uploaders]):
        indices = [idx for idx, label in enumerate(data_y) if label.split('.')[0] in labels]
        selected_X = data_x[indices]
        selected_y = data_y[indices].codes

        X_save_dir = os.path.join(data_save_root, "uploader_%d_X.pkl" % (i))
        y_save_dir = os.path.join(data_save_root, "uploader_%d_y.pkl" % (i))

        with open(X_save_dir, "wb") as f:
            pickle.dump(selected_X, f)
        with open(y_save_dir, "wb") as f:
            pickle.dump(selected_y, f)
        print("Saving to %s" % (X_save_dir))

# 随机选取
# def generate_user(data_x, data_y, n_users=50, data_save_root=None):
#     if data_save_root is None:
#         return
#     os.makedirs(data_save_root, exist_ok=True)
#     n = len(data_x)
#     for i in range(n_users):
#         selected_X = data_x[i * (n // n_users): (i + 1) * (n // n_users)]
#         selected_y = data_y[i * (n // n_users): (i + 1) * (n // n_users)].codes
#         X_save_dir = os.path.join(data_save_root, "user_%d_X.pkl" % (i))
#         y_save_dir = os.path.join(data_save_root, "user_%d_y.pkl" % (i))
#         with open(X_save_dir, "wb") as f:
#             pickle.dump(selected_X, f)
#         with open(y_save_dir, "wb") as f:
#             pickle.dump(selected_y, f)
#         print("Saving to %s" % (X_save_dir))

def generate_user(data_x, data_y, n_users=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)
    n = len(data_x)
    for i, labels in enumerate(super_classes_select3[:n_users]):
        indices = [idx for idx, label in enumerate(data_y) if label.split('.')[0] in labels]
        selected_X = data_x[indices]
        selected_y = data_y[indices].codes

        X_save_dir = os.path.join(data_save_root, "user_%d_X.pkl" % (i))
        y_save_dir = os.path.join(data_save_root, "user_%d_y.pkl" % (i))
        with open(X_save_dir, "wb") as f:
            pickle.dump(selected_X, f)
        with open(y_save_dir, "wb") as f:
            pickle.dump(selected_y, f)
        print("Saving to %s" % (X_save_dir))


# Train Uploaders' models
def train(X, y, out_classes):
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_tfidf, y)

    return vectorizer, clf


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
