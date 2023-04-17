import gc
import joblib
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, warnings
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel


from .utils import *
from .config import model_dir, grid_dir, store_list, lgb_params_list

warnings.filterwarnings("ignore")


def train_lgb_model(train_x, train_y, val_x, val_y, store, lr, nl, md, best, save=True, n_estimators=0, train_flag=0):
    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "rmse",
        "metric": "rmse",
        "learning_rate": lr,
        "num_leaves": nl,
        "max_depth": md,
        "n_estimators": 100000,
        "boost_from_average": False,
        "verbose": -1,
    }

    if train_flag:
        idx = int(len(train_y) * 0.1)
        train_data = lgb.Dataset(train_x[:-idx], label=train_y[:-idx])
        val_data = lgb.Dataset(train_x[-idx:], label=train_y[-idx:])
    else:
        train_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(val_x, label=val_y)

    if n_estimators:
        lgb_params["n_estimators"] = n_estimators
        gbm = lgb.train(lgb_params, train_data, verbose_eval=100)
    else:
        gbm = lgb.train(lgb_params, train_data, valid_sets=[val_data], verbose_eval=100, early_stopping_rounds=1000)

    test_y = gbm.predict(val_x, num_iteration=gbm.best_iteration)
    res = mean_squared_error(val_y, test_y, squared=False)

    if res < best:
        best = res
        if save:
            gbm.save_model(os.path.join(model_dir, f"lgb_{store}.out"))

    return best


def train_ridge_model(train_x, train_y, val_x, val_y, store, a, best, save=True):
    model = Ridge(alpha=a)
    model.fit(train_x, train_y)

    test_y = model.predict(val_x)
    res = mean_squared_error(val_y, test_y, squared=False)

    if res < best:
        best = res
        if save:
            joblib.dump(model, os.path.join(model_dir, f"ridge_{store}.out"))

    return best


def train_svm_model(
    train_x, train_y, val_x, val_y, store, C, epsilon, best, save=True, gamma=0.1, adaptation_model=[], K1=None, K2=None
):
    if K1 is None:
        model = SVR(C=C, epsilon=epsilon, max_iter=30000, cache_size=10240, verbose=True, gamma=gamma)
    else:
        model = AuxiliarySVR(
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            adaptation_model=adaptation_model,
            max_iter=30000,
            cache_size=10240,
            verbose=True,
            K1=K1,
            K2=K2,
        )

    model.fit(train_x, train_y)
    test_y = model.predict(val_x)
    res = mean_squared_error(val_y, test_y, squared=False)

    if res < best:
        best = res
        if save:
            joblib.dump(model, os.path.join(model_dir, f"svm_{store}.out"))

    return best


def train_krr_model(train_x, train_y, val_x, val_y, store, a, best, save=True, gamma=0.1, K1=None, K2=None):
    if K1 is None:
        model = KernelRidge(kernel="rbf", alpha=a, gamma=gamma)
        model.fit(train_x, train_y)
        test_y = model.predict(val_x)
        res = mean_squared_error(val_y, test_y, squared=False)
    else:
        len1, len2 = len(train_y), len(val_y)
        model = KernelRidge(kernel="precomputed", alpha=a)
        model.fit(K1[-len1:, -len1:], train_y)
        test_y = model.predict(K2[-len2:, -len1:])
        res = mean_squared_error(val_y, test_y, squared=False)

    if res < best:
        best = res
        if save:
            joblib.dump(model, os.path.join(model_dir, f"krr_{store}.out"))

    return best


def grid_search(store_id, algo, search_lgb_flag=False):
    store = store_list[store_id]

    if algo == "lgb":
        train_x, train_y, val_x, val_y = acquire_data(store, True)
        learning_rate = [0.005, 0.01, 0.015]
        num_leaves = [128, 224, 300]
        max_depth = [50, 66, 80]
        best = 10000000

        if search_lgb_flag:
            for lr in learning_rate:
                for nl in num_leaves:
                    for md in max_depth:
                        best = train_lgb_model(train_x, train_y, val_x, val_y, store, lr, nl, md, best)
                        print(f"store: {store}, lr: {lr}, nl: {nl}, md: {md}, best: {best}")
        else:
            lr, nl, md = lgb_params_list[store_id]
            best = train_lgb_model(train_x, train_y, val_x, val_y, store, lr, nl, md, best)
            print(f"store: {store}, lr: {lr}, nl: {nl}, md: {md}, best: {best}")
    elif algo == "ridge":
        train_x, train_y, val_x, val_y = acquire_data(store, True)
        alpha = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 30]
        best = 10000000

        for a in alpha:
            best = train_ridge_model(train_x, train_y, val_x, val_y, store, a, best)
            print(f"store: {store}, alpha: {a}, best: {best}")


def grid_training_sample(algo, user_list=list(range(10))):
    for i in range(len(user_list)):
        store_id = user_list[i]
        store = store_list[store_id]
        org_train_x, org_train_y, val_x, val_y = acquire_data(store, True)
        res = []

        proportion_list = [
            100,
            300,
            500,
            700,
            900,
            1000,
            3000,
            5000,
            7000,
            9000,
            10000,
            30000,
            50000,
            70000,
            90000,
            100000,
            300000,
            500000,
            700000,
            900000,
            1000000,
            3000000,
            5000000,
        ]

        for proportion in proportion_list:
            """
            random
            org_idx_list = list(range(len(org_train_y)))
            idx_list = random.sample(org_idx_list, min(proportion, len(org_train_y)))
            train_x = org_train_x.iloc[idx_list]
            train_y = org_train_y.iloc[idx_list]
            """
            train_x = org_train_x[-proportion:]
            train_y = org_train_y[-proportion:]
            best = 10000000

            if algo == "lgb":
                lr, nl, md = lgb_params_list[store_id]
                best = train_lgb_model(
                    train_x, train_y, val_x, val_y, store, lr, nl, md, best, save=False, n_estimators=3000, train_flag=0
                )
                print(f"store: {store}, lr: {lr}, nl: {nl}, md: {md}, best: {best}")

            elif algo == "ridge":
                alpha = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 30]
                for a in alpha:
                    best = train_ridge_model(train_x, train_y, val_x, val_y, store, a, best, save=False)
                    print(f"store: {store}, alpha: {a}, best: {best}")

            elif algo == "svm":
                C = [1, 10, 100]
                epsilon = 0.001
                for c in C:
                    best = train_svm_model(train_x, train_y, val_x, val_y, store, c, epsilon, best, save=False)
                    print(f"store: {store}, C: {c}, epsilon: {epsilon}, best: {best}")

            res.append([proportion, best])
            np.savetxt(os.path.join(grid_dir, f"grid_sample_{algo}_{store}.out"), np.array(res))

            if proportion > len(org_train_y):
                break


def retrain_models(algo):
    for store_id in range(10):
        grid_search(store_id, algo)


def train_adaptation_grid(
    algo, max_sample, test_sample, user_list=list(range(10)), adaptation_model=[], residual=False
):
    """
    adaptation_model = [
        [("lgb", 1), ("ridge", 2)],
        [("lgb", 1), ("ridge", 2)]
    ]
    """

    proportion_list = [
        100,
        300,
        500,
        700,
        900,
        1000,
        3000,
        5000,
        7000,
        9000,
        10000,
        30000,
        50000,
        70000,
        90000,
        100000,
        300000,
        500000,
        700000,
        900000,
        1000000,
        3000000,
        5000000,
    ]
    sample_idx = proportion_list.index(max_sample) + 1

    for i in range(len(user_list)):
        store_id = user_list[i]
        store = store_list[store_id]
        org_train_x, org_train_y, val_x, val_y = acquire_data(store, True)
        val_x = val_x[-test_sample:]
        val_y = val_y[-test_sample:]

        if algo == "lgb" or algo == "ridge":
            res = []

            if adaptation_model != []:
                if residual:
                    aux_algo, model_idx = adaptation_model[i][0]
                    org_train_y -= model_predict(aux_algo, model_idx, org_train_x)
                    val_y -= model_predict(aux_algo, model_idx, val_x)

                else:
                    train_y_list, val_y_list = [], []

                    for aux_algo, model_idx in adaptation_model[i]:
                        train_y_list.append(model_predict(aux_algo, model_idx, org_train_x))
                        val_y_list.append(model_predict(aux_algo, model_idx, val_x))

                    for j in range(len(train_y_list)):
                        org_train_x[f"model_values_{j}"] = train_y_list[j]
                        val_x[f"model_values_{j}"] = val_y_list[j]

            for proportion in proportion_list[:sample_idx]:
                """
                random
                org_idx_list = list(range(len(org_train_y)))
                idx_list = random.sample(org_idx_list, min(proportion, len(org_train_y)))
                train_x = org_train_x.iloc[idx_list]
                train_y = org_train_y.iloc[idx_list]
                """
                train_x = org_train_x[-proportion:]
                train_y = org_train_y[-proportion:]
                best = 10000000

                if algo == "lgb":
                    if max_sample < 50000:
                        learning_rate = [0.005, 0.01, 0.015]
                        num_leaves = [128, 224, 300]
                        max_depth = [50, 66, 80]

                        for lr in learning_rate:
                            for nl in num_leaves:
                                for md in max_depth:
                                    best = train_lgb_model(
                                        train_x, train_y, val_x, val_y, store, lr, nl, md, best, save=False
                                    )
                                    print(f"store: {store}, lr: {lr}, nl: {nl}, md: {md}, best: {best}")
                    else:
                        lr, nl, md = lgb_params_list[store_id]
                        best = train_lgb_model(
                            train_x,
                            train_y,
                            val_x,
                            val_y,
                            store,
                            lr,
                            nl,
                            md,
                            best,
                            save=False,
                            n_estimators=3000,
                            train_flag=0,
                        )
                        print(f"store: {store}, lr: {lr}, nl: {nl}, md: {md}, best: {best}")

                elif algo == "ridge":
                    alpha = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 30]
                    for a in alpha:
                        best = train_ridge_model(train_x, train_y, val_x, val_y, store, a, best, save=False)
                        print(f"store: {store}, alpha: {a}, best: {best}")

                res.append([proportion, best])
                text = str(adaptation_model[i]) if adaptation_model != [] else "null"
                text += "_residual_" if residual else ""
                np.savetxt(os.path.join(grid_dir, f"{algo}_using_{text}_{store}.out"), np.array(res))

                if proportion > len(org_train_y):
                    break

        elif algo == "svm" or algo == "krr":
            res = [[proportion, 10000] for proportion in proportion_list[:sample_idx]]
            org_train_x = org_train_x.to_numpy()
            org_train_y = org_train_y.to_numpy()
            val_x = val_x.to_numpy()
            val_y = val_y.to_numpy()

            y1_list, y2_list = [], []
            gamma_list = [0.01, 0.1, 0.5, 1]

            if residual:
                aux_algo, model_idx = adaptation_model[i][0]
                org_train_y = org_train_y.astype(np.float64)
                val_y = val_y.astype(np.float64)
                org_train_y -= model_predict(aux_algo, model_idx, org_train_x)
                val_y -= model_predict(aux_algo, model_idx, val_x)

            elif adaptation_model != []:
                for aux_algo, idx in adaptation_model[i]:
                    y1_list.append(model_predict(aux_algo, idx, org_train_x[-max_sample:]).reshape(-1, 1))
                    y2_list.append(model_predict(aux_algo, idx, val_x).reshape(-1, 1))

            for gamma in gamma_list:
                K1 = np.zeros((max_sample, max_sample))
                K2 = np.zeros((len(val_x), max_sample))

                if (not residual) and adaptation_model != []:
                    for j in range(len(adaptation_model[i])):
                        aux_algo, idx = adaptation_model[i][j]
                        y1 = y1_list[j]
                        y2 = y2_list[j]
                        K1 += np.dot(y1, y1.T)
                        K2 += np.dot(y2, y1.T)

                K1 += rbf_kernel(org_train_x[-max_sample:], org_train_x[-max_sample:], gamma=gamma)
                K2 += rbf_kernel(val_x, org_train_x[-max_sample:], gamma=gamma)

                for idx in range(len(proportion_list[:sample_idx])):
                    proportion = proportion_list[idx]
                    """
                    random
                    org_idx_list = list(range(len(org_train_y)))
                    idx_list = random.sample(org_idx_list, min(proportion, len(org_train_y)))
                    train_x = org_train_x.iloc[idx_list]
                    train_y = org_train_y.iloc[idx_list]
                    """
                    train_x = org_train_x[-proportion:]
                    train_y = org_train_y[-proportion:]
                    best = 10000000

                    if algo == "svm":
                        C = [1, 10, 50, 100, 200]
                        epsilon = 0.001

                        for c in C:
                            adapt_m = [] if adaptation_model == [] else adaptation_model[i]
                            best = train_svm_model(
                                train_x,
                                train_y,
                                val_x,
                                val_y,
                                store,
                                c,
                                epsilon,
                                best,
                                save=False,
                                gamma=gamma,
                                adaptation_model=adapt_m,
                                K1=K1,
                                K2=K2,
                            )
                            print(f"store: {store}, gamma: {gamma}, C: {c}, epsilon: {epsilon}, best: {best}")

                    elif algo == "krr":
                        alpha = [0.01, 0.1, 0.5, 1.0, 5.0, 10]

                        for a in alpha:
                            best = train_krr_model(
                                train_x, train_y, val_x, val_y, store, a, best, save=False, gamma=gamma, K1=K1, K2=K2
                            )
                            print(f"store: {store}, a: {a}, gamma: {gamma}, best: {best}")

                    if best < res[idx][1]:
                        res[idx][1] = best
                    text = str(adaptation_model[i]) if adaptation_model != [] else "null"
                    text += "_residual" if residual else ""
                    np.savetxt(os.path.join(grid_dir, f"{algo}_using_{text}_{store}.out"), np.array(res))

                    if proportion > len(org_train_y):
                        break

                    del train_x, train_y
                    gc.collect()

                del K1, K2
                gc.collect()

            del org_train_x, org_train_y
            gc.collect()
