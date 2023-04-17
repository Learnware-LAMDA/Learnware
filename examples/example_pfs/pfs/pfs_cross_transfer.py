import os
import pickle
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.seterr(divide="ignore", invalid="ignore")
from .paths import pfs_split_dir, pfs_res_dir, model_dir

np.random.seed(0)


def load_pfs_data(fpath):
    df = pd.read_csv(fpath)

    features = list(df.columns)
    features.remove("item_cnt_month")
    features.remove("date_block_num")

    # remove id info
    # features.remove('shop_id')
    # features.remove('item_id')

    # remove discrete info
    # features.remove('city_code')
    # features.remove('item_category_code')
    # features.remove('item_category_common')

    xs = df[features].values
    ys = df["item_cnt_month"].values

    categorical_feature_names = ["country_part", "item_category_common", "item_category_code", "city_code"]
    types = None

    return xs, ys, features, types


def get_split_errs(algo):
    """
    according to proportion_list, generate errs whose shape is [shop, split_data]
    """
    shop_ids = [i for i in range(60) if i not in [0, 1, 40]]
    shop_ids = [i for i in shop_ids if i not in [8, 11, 23, 36]]
    user_list = [i for i in range(53)]
    proportion_list = [100, 300, 500, 700, 900, 1000, 3000, 5000, 7000, 9000, 10000, 30000, 50000, 70000]

    # train
    errs = np.zeros((len(user_list), len(proportion_list)))
    for s, sid in enumerate(user_list):
        # load train data
        fpath = os.path.join(pfs_split_dir, "Shop{:0>2d}-train.csv".format(shop_ids[sid]))
        fpath_val = os.path.join(pfs_split_dir, "Shop{:0>2d}-val.csv".format(shop_ids[sid]))
        train_xs, train_ys, _, _ = load_pfs_data(fpath)
        val_xs, val_ys, _, _ = load_pfs_data(fpath_val)
        print(shop_ids[sid], train_xs.shape, train_ys.shape)
        # data regu
        # train_xs = (train_xs - train_xs.min(0)) / (train_xs.max(0) - train_xs.min(0) + 0.0001)
        # val_xs = (val_xs - val_xs.min(0)) / (val_xs.max(0) - val_xs.min(0) + 0.0001)

        if algo == "lgb":
            for tmp in range(len(proportion_list)):
                model = lgb.LGBMModel(
                    boosting_type="gbdt",
                    num_leaves=2 ** 7 - 1,
                    learning_rate=0.01,
                    objective="rmse",
                    metric="rmse",
                    feature_fraction=0.75,
                    bagging_fraction=0.75,
                    bagging_freq=5,
                    seed=1,
                    verbose=1,
                    n_estimators=100000,
                )
                model_ori = joblib.load(os.path.join(model_dir, "{}_Shop{:0>2d}.out".format("lgb", shop_ids[sid])))
                para = model_ori.get_params()
                para["n_estimators"] = 1000
                model.set_params(**para)
                split = train_xs.shape[0] - proportion_list[tmp]

                model.fit(
                    train_xs[split:,],
                    train_ys[split:],
                    eval_set=[(val_xs, val_ys)],
                    early_stopping_rounds=50,
                    verbose=100,
                )
                pred_ys = model.predict(val_xs)
                rmse = np.sqrt(((val_ys - pred_ys) ** 2).mean())
                errs[s][tmp] = rmse
    return errs


def get_errors(algo):
    shop_ids = [i for i in range(60) if i not in [0, 1, 40]]
    shop_ids = [i for i in shop_ids if i not in [8, 11, 23, 36]]

    # train
    K = len(shop_ids)

    feature_weight = np.zeros(())
    errs = np.zeros((K, K))
    for s, sid in enumerate(shop_ids):
        # load train data
        fpath = os.path.join(pfs_split_dir, "Shop{:0>2d}-train.csv".format(sid))
        fpath_val = os.path.join(pfs_split_dir, "Shop{:0>2d}-val.csv".format(sid))
        train_xs, train_ys, features, _ = load_pfs_data(fpath)
        val_xs, val_ys, _, _ = load_pfs_data(fpath_val)
        print(sid, train_xs.shape, train_ys.shape)
        if s == 0:
            feature_weight = np.zeros((K, len(features)))

        if algo == "lgb":
            model = lgb.LGBMModel(
                boosting_type="gbdt",
                num_leaves=2 ** 7 - 1,
                learning_rate=0.01,
                objective="rmse",
                metric="rmse",
                feature_fraction=0.75,
                bagging_fraction=0.75,
                bagging_freq=5,
                seed=1,
                verbose=1,
                n_estimators=1000,
            )
            # train regu data
            # train_xs = (train_xs - train_xs.min(0)) / (train_xs.max(0) - train_xs.min(0) + 0.0001)
            # val_xs = (val_xs - val_xs.min(0)) / (val_xs.max(0) - val_xs.min(0) + 0.0001)
            model.fit(train_xs, train_ys, eval_set=[(val_xs, val_ys)], early_stopping_rounds=100, verbose=100)

            # grid search
            # para = {'learning_rate': [0.005, 0.01, 0.015], 'num_leaves' : [128, 224, 300], 'max_depth' : [50, 66, 80]}
            # grid_search = GridSearchCV(model, para, scoring='neg_mean_squared_error')
            # grid_result = grid_search.fit(train_xs, train_ys, eval_set=[(val_xs, val_ys)], verbose = 1000, early_stopping_rounds=1000)
            # model = grid_result.best_estimator_

            joblib.dump(model, os.path.join(model_dir, "{}_Shop{:0>2d}.out".format(algo, sid)))

            importances = model.feature_importances_
        elif algo == "ridge":
            # train_xs = (train_xs - train_xs.min(0)) / (train_xs.max(0) - train_xs.min(0) + 0.0001)
            model = Ridge()

            para = {"alpha": [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 30]}
            grid_search = GridSearchCV(model, para)
            grid_result = grid_search.fit(train_xs, train_ys)

            model = grid_result.best_estimator_
            importances = model.coef_
            joblib.dump(model, os.path.join(model_dir, "{}_Shop{:0>2d}.out".format(algo, sid)))

        feature_weight[s] = importances
        # leave one out test
        for t, tid in enumerate(shop_ids):
            # load test data
            fpath = os.path.join(pfs_split_dir, "Shop{:0>2d}-val.csv".format(tid))
            test_xs, test_ys, _, _ = load_pfs_data(fpath)
            # data regu
            # test_xs = (test_xs - test_xs.min(0)) / (test_xs.max(0) - test_xs.min(0) + 0.0001)

            pred_ys = model.predict(test_xs)

            rmse = np.sqrt(((test_ys - pred_ys) ** 2).mean())

            print("Shop{} --> Shop{}: {}".format(s, t, rmse))

            errs[s][t] = rmse
    np.savetxt(os.path.join(pfs_res_dir, "PFS_{}_weights.txt".format(algo)), feature_weight)
    return errs


def plot_heatmap(mat, algo):
    x_labels = [f"Model{i}" for i in range(mat.shape[1])]
    y_labels = [f"Task{i}" for i in range(mat.shape[0])]

    fig = plt.figure(figsize=(10, 9))
    plt.subplot(1, 1, 1)
    ax = plt.gca()
    im = plt.imshow(mat)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.3)
    plt.colorbar(im, cax=cax)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))

    ax.set_title(f"RMSE on Test set ({algo})")
    plt.tight_layout()
    plt.savefig(os.path.join(pfs_res_dir, "PFS_{}_heatmap.jpg".format(algo)), dpi=700)


def plot_var(errs, algo):
    avg_err = []
    min_err = []
    med_err = []
    max_err = []
    std_err = []
    cnts = []
    improves = []

    for j in range(len(errs)):
        inds = [i for i in range(len(errs)) if i != j]
        ys = errs[:, j][inds]
        avg_err.append(np.mean(ys))
        min_err.append(np.min(ys))
        med_err.append(np.median(ys))
        max_err.append(np.max(ys))
        std_err.append(np.std(ys))
        cnts.append(np.sum(ys >= np.mean(ys)))
        improves.append((np.mean(ys) - np.min(ys)) / np.mean(ys))

    avg_err = np.array(avg_err)
    min_err = np.array(min_err)
    med_err = np.array(med_err)
    max_err = np.array(max_err)
    std_err = np.array(std_err)
    cnts = np.array(cnts)
    improves = np.array(improves)

    inds = np.argsort(avg_err)

    avg_err = avg_err[inds]
    min_err = min_err[inds]
    med_err = med_err[inds]
    max_err = max_err[inds]
    std_err = std_err[inds]
    cnts = cnts[inds]
    improves = improves[inds]
    xs = list(range(len(inds)))

    fig = plt.figure(figsize=(8, 8))

    ax = plt.subplot(3, 1, 1)
    ax.plot(xs, avg_err, color="red", linestyle="solid", linewidth=2.5)
    ax.plot(xs, min_err, color="blue", linestyle="dotted", linewidth=1.5)
    ax.plot(xs, med_err, color="purple", linestyle="solid", linewidth=1.0)
    ax.plot(xs, max_err, color="green", linestyle="dashed", linewidth=1.5)

    ax.legend(["Avg", "Min", "Median", "Max"], fontsize=14)

    ax.fill_between(xs, avg_err - std_err, avg_err + std_err, alpha=0.2)

    gap = np.mean(avg_err - min_err)

    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_title("RMSE of Source Models ({}) [Avg-Min:{:.3f}]".format(algo, gap), fontsize=18)

    ax = plt.subplot(3, 1, 2)
    ax.bar(xs, cnts)
    ax.set_ylabel("Number", fontsize=14)
    ax.set_title("Number of sources above average", fontsize=18)

    ax = plt.subplot(3, 1, 3)
    ax.plot(xs, improves)
    ax.set_xlabel("Sorted Shop ID by Avg.Err", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.set_title("Best Improve Ratio: (Avg - Min) / Avg", fontsize=18)

    fig.tight_layout()
    fig.savefig(os.path.join(pfs_res_dir, "{}-var.jpg".format(algo)))
    plt.show()


def plot_performance(errs, weights, algo):
    avg_err = []
    min_err = []
    med_err = []
    max_err = []
    std_err = []
    cnts = []
    improves = []

    for i in range(errs.shape[0]):
        inds = [j for j in range(errs.shape[1]) if j != i]
        arr = errs[i][inds]
        avg_err.append(np.mean(arr))
        min_err.append(np.min(arr))
        med_err.append(np.median(arr))
        max_err.append(np.max(arr))
        std_err.append(np.std(arr))
        cnts.append(np.sum(arr >= np.mean(arr)))
        improves.append((np.mean(arr) - np.min(arr)) / np.mean(arr))

    avg_err = np.array(avg_err)
    min_err = np.array(min_err)
    med_err = np.array(med_err)
    max_err = np.array(max_err)
    std_err = np.array(std_err)
    cnts = np.array(cnts)
    improves = np.array(improves)

    inds = np.argsort(avg_err)
    avg_err = avg_err[inds]
    min_err = min_err[inds]
    med_err = med_err[inds]
    max_err = max_err[inds]
    std_err = std_err[inds]
    cnts = cnts[inds]
    improves = improves[inds]
    xs = list(range(len(inds)))

    fig = plt.figure(figsize=(12, 9))

    ax = plt.subplot(2, 2, 1)
    ax.plot(xs, avg_err, color="red", linestyle="solid", linewidth=2.5)
    ax.plot(xs, min_err, color="blue", linestyle="dotted", linewidth=1.5)
    ax.plot(xs, med_err, color="purple", linestyle="solid", linewidth=1.0)
    ax.plot(xs, max_err, color="green", linestyle="dashed", linewidth=1.5)

    ax.legend(["Avg", "Min", "Median", "Max"], fontsize=14)

    ax.fill_between(xs, avg_err - std_err, avg_err + std_err, alpha=0.2)

    gap = np.mean(avg_err - min_err)

    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_title("RMSE of Source Models ({}) [Avg-Min:{:.3f}]".format(algo, gap), fontsize=18)

    ax = plt.subplot(2, 2, 2)
    ax.bar(xs, cnts)
    ax.set_ylabel("Number", fontsize=14)
    ax.set_title("Number of sources above average", fontsize=18)

    ax = plt.subplot(2, 2, 3)
    ax.plot(xs, improves)
    ax.set_xlabel("Sorted Shop ID by Avg.Err", fontsize=14)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.set_title("Best Improve Ratio: (Avg - Min) / Avg", fontsize=18)

    ax = plt.subplot(2, 2, 4)
    weights = np.mean(weights, axis=0) / weights.sum()
    weights = np.sort(weights)
    xs = list(range(len(weights)))
    ax.plot(xs, weights)
    # ax.set_xlabel("Sorted Feature ID by Avg.Feature_Importance", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.set_title("Avg.Feature_Importances", fontsize=18)

    fig.tight_layout()
    fig.savefig(os.path.join(pfs_res_dir, "PFS_{}_performance.png".format(algo)), dpi=700)
    # fig.savefig(f"{algo}_performance.png", dpi=700)
    plt.show()


if __name__ == "__main__":
    # for algo in ["ridge", "lgb", "xgboost_125"]:
    for algo in ["ridge"]:
        fpath = os.path.join(pfs_res_dir, "{}_errs.pkl".format(algo))
        if os.path.exists(fpath):
            with open(fpath, "rb") as fr:
                errs = pickle.load(fr)
        else:
            errs = get_errors(algo=algo)
            with open(fpath, "wb") as fw:
                pickle.dump(errs, fw)

        index = ["Source{}".format(k) for k in range(len(errs))]
        columns = ["Target{}".format(k) for k in range(len(errs[0]))]
        df = pd.DataFrame(errs, index=index, columns=columns)

        fpath = os.path.join(pfs_res_dir, "PFS_{}_errs.txt".format(algo))
        # df.to_csv(fpath, index=True)
        np.savetxt(fpath, errs.T)

        # plot_var(errs, algo)
        plot_heatmap(errs.T, algo)
        weights = np.loadtxt(os.path.join(pfs_res_dir, "PFS_{}_weights.txt".format(algo)))
        plot_performance(errs.T, weights, algo)
