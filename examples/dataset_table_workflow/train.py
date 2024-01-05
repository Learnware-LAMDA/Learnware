import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_squared_error

from learnware.logger import get_module_logger
from config import user_model_params

logger = get_module_logger("train_table", level="INFO")


def train_lgb(X_train, y_train, X_val, y_val, dataset):
    model_param = user_model_params[dataset]["lgb"]
    params = model_param["params"]

    MAX_ROUNDS = model_param["MAX_ROUNDS"]
    val_pred = []
    cate_vars = []

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cate_vars)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cate_vars)
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval] if dataset == "Corporacion" else [dval],
        callbacks=[early_stopping(model_param["early_stopping_rounds"], verbose=False)]
    )
    val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    return bst


def train_ridge(X_train, y_train, X_val, y_val, dataset):
    pass


def train_model(X_train, y_train, X_val, y_val, test_info):
    dataset = test_info["dataset"]
    model_type = test_info["model_type"]
    assert model_type in ["lgb", "ridge"]
    
    if model_type == "lgb":
        return train_lgb(X_train, y_train, X_val, y_val, dataset)