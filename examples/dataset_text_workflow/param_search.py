import json
import pickle

import joblib
import numpy as np
import os, warnings
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK

warnings.filterwarnings("ignore")


def train_lgb(params, train_x, val_x, train_y, val_y, num_class, classification, model_path, save=False):
    train_data = lgb.Dataset(train_x, label=train_y)
    val_data = lgb.Dataset(val_x, label=val_y)

    if classification:
        lgb_params = {
            **params,
            # "boosting": "dart",
            "learning_rate": 0.1,
            "importance_type": "gain",
            # "class_weight": 'balanced',
            "objective": "multiclass",
            "num_class": num_class,
            "n_estimators": 1000,
            "early_stopping_round": 30,
            'max_bin': 512,
            "verbose": -1,
        }
    else:
        lgb_params = {
            **params,
            # "boosting": "dart",
            "learning_rate": 0.1,
            "importance_type": "gain",
            "objective": "regression",
            "n_estimators": 1000,
            "early_stopping_round": 30,
            'max_bin': 512,
            "verbose": -1,
        }
    gbm = lgb.train(lgb_params, train_data, valid_sets=[val_data])
    pred_val_y = gbm.predict(val_x, num_iteration=gbm.best_iteration)
    if classification:
        pred_val_y = np.argmax(pred_val_y, 1)
        res = 1 - accuracy_score(val_y, pred_val_y)
    else:
        res = mean_squared_error(val_y, pred_val_y, squared=False)

    pred_train_y = gbm.predict(train_x, num_iteration=gbm.best_iteration)
    if classification:
        pred_train_y = np.argmax(pred_train_y, 1)
        res_train = 1 - accuracy_score(train_y, pred_train_y)
    else:
        res_train = mean_squared_error(train_y, pred_train_y, squared=False)

    # print(params, res)
    if save:
        # gbm.save_model(model_path)
        # with open(model_path, 'wb') as f:
        #     pickle.dump(gbm, f)
        print(params, res_train, res)
    return {'loss': res, 'status': STATUS_OK, 'param': params}


def grid_search(train_x, val_x, train_y, val_y, num_class, classification, model_path):
    def objective(params):
        return train_lgb(params, train_x, val_x, train_y, val_y, num_class, classification, model_path, save=False)

    if classification:
        space = {
            'max_depth': hp.choice('max_depth', [3, 4, 5]),
            'num_leaves': hp.choice('num_leaves', [5, 6, 7, 12, 13, 14, 15, 28, 29, 30, 31]),
            'subsample': hp.choice('subsample', [0.6, 0.8, 0.9, 1.0]),
            'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.8, 0.9, 1.0]),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(1000)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(1000)),
            'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 20, 50]),
        }
    else:
        space = {
            'max_depth': hp.choice('max_depth', [3, 4, 5]),
            'num_leaves': hp.choice('num_leaves', [5, 6, 7, 12, 13, 14, 15, 28, 29, 30, 31]),
            'subsample': hp.choice('subsample', [0.6, 0.8, 0.9, 1.0]),
            'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.8, 0.9, 1.0]),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(1000)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(1000))
        }
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)
    best_params = trials.best_trial['result']['param']
    res = train_lgb(best_params, train_x, val_x, train_y, val_y, num_class, classification, model_path, save=True)[
        'param']
    return res


if __name__ == "__main__":
    X_path = os.path.join("data/processed_data/ae/uploader/uploader_0_X.pkl")
    y_path = os.path.join("data/processed_data/ae/uploader/uploader_0_y.pkl")
    with open(X_path, "rb") as f:
        X = pickle.load(f)
    with open(y_path, "rb") as f:
        y = pickle.load(f)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words="english")
    X_trian_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    res = grid_search(X_trian_tfidf, X_valid_tfidf, y_train, y_valid, 3, True, "models/model.txt")
