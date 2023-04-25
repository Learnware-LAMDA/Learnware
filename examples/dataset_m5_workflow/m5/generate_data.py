import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from copy import deepcopy as dco
import os, sys, gc, time, warnings, pickle, psutil, random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from .utils import *
from .config import raw_data_dir, processed_data_dir, TARGET

warnings.filterwarnings("ignore")


# ==================== preprocessing ====================
def melt_raw_data(train_df):
    if os.path.exists(os.path.join(processed_data_dir, "melt_raw_data.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "melt_raw_data.pkl"))

    index_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    grid_df = pd.melt(train_df, id_vars=index_columns, var_name="d", value_name=TARGET)

    for col in index_columns:
        grid_df[col] = grid_df[col].astype("category")

    grid_df.to_pickle(os.path.join(processed_data_dir, "melt_raw_data.pkl"))
    return grid_df


def add_release_week(grid_df, prices_df, calendar_df):
    if os.path.exists(os.path.join(processed_data_dir, "add_release_week.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "add_release_week.pkl"))

    release_df = prices_df.groupby(["store_id", "item_id"])["wm_yr_wk"].agg(["min"]).reset_index()
    release_df.columns = ["store_id", "item_id", "release"]
    grid_df = merge_by_concat(grid_df, release_df, ["store_id", "item_id"])
    grid_df = merge_by_concat(grid_df, calendar_df[["wm_yr_wk", "d"]], ["d"])

    # cutoff meaningless rows
    grid_df = grid_df[grid_df["wm_yr_wk"] >= grid_df["release"]]
    grid_df = grid_df.reset_index(drop=True)

    # scale the release
    grid_df["release"] = grid_df["release"] - grid_df["release"].min()
    grid_df["release"] = grid_df["release"].astype(np.int16)

    grid_df.to_pickle(os.path.join(processed_data_dir, "add_release_week.pkl"))
    return grid_df


def add_prices(grid_df, prices_df, calendar_df):
    if os.path.exists(os.path.join(processed_data_dir, "add_prices.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "add_prices.pkl"))

    prices_df["price_max"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("max")
    prices_df["price_min"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("min")
    prices_df["price_std"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("std")
    prices_df["price_mean"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("mean")
    prices_df["price_norm"] = prices_df["sell_price"] / prices_df["price_max"]

    prices_df["price_nunique"] = prices_df.groupby(["store_id", "item_id"])["sell_price"].transform("nunique")
    prices_df["item_nunique"] = prices_df.groupby(["store_id", "sell_price"])["item_id"].transform("nunique")

    calendar_prices = calendar_df[["wm_yr_wk", "month", "year"]]
    calendar_prices = calendar_prices.drop_duplicates(subset=["wm_yr_wk"])
    prices_df = prices_df.merge(calendar_prices[["wm_yr_wk", "month", "year"]], on=["wm_yr_wk"], how="left")

    prices_df["price_momentum"] = prices_df["sell_price"] / prices_df.groupby(["store_id", "item_id"])[
        "sell_price"
    ].transform(lambda x: x.shift(1))
    prices_df["price_momentum_m"] = prices_df["sell_price"] / prices_df.groupby(["store_id", "item_id", "month"])[
        "sell_price"
    ].transform("mean")
    prices_df["price_momentum_y"] = prices_df["sell_price"] / prices_df.groupby(["store_id", "item_id", "year"])[
        "sell_price"
    ].transform("mean")

    grid_df = reduce_mem_usage(grid_df)
    prices_df = reduce_mem_usage(prices_df)

    original_columns = list(grid_df)
    grid_df = grid_df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    grid_df = reduce_mem_usage(grid_df)

    grid_df.to_pickle(os.path.join(processed_data_dir, "add_prices.pkl"))
    return grid_df


def add_date(grid_df, calendar_df):
    if os.path.exists(os.path.join(processed_data_dir, "add_date.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "add_date.pkl"))

    # merge calendar partly
    icols = [
        "date",
        "d",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    grid_df = grid_df.merge(calendar_df[icols], on=["d"], how="left")

    # convert to category
    icols = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    for col in icols:
        grid_df[col] = grid_df[col].astype("category")

    # make some features from date
    grid_df["date"] = pd.to_datetime(grid_df["date"])
    grid_df["tm_d"] = grid_df["date"].dt.day.astype(np.int8)
    grid_df["tm_w"] = grid_df["date"].dt.week.astype(np.int8)
    grid_df["tm_m"] = grid_df["date"].dt.month.astype(np.int8)
    grid_df["tm_y"] = grid_df["date"].dt.year
    grid_df["tm_y"] = (grid_df["tm_y"] - grid_df["tm_y"].min()).astype(np.int8)
    grid_df["tm_wm"] = grid_df["tm_d"].apply(lambda x: ceil(x / 7)).astype(np.int8)

    grid_df["tm_dw"] = grid_df["date"].dt.dayofweek.astype(np.int8)
    grid_df["tm_w_end"] = (grid_df["tm_dw"] >= 5).astype(np.int8)

    # clear columns
    grid_df["d"] = grid_df["d"].apply(lambda x: x[2:]).astype(np.int16)
    grid_df = grid_df.drop("wm_yr_wk", 1)

    grid_df.to_pickle(os.path.join(processed_data_dir, "add_date.pkl"))
    return grid_df


def add_lags_rollings(grid_df):
    if os.path.exists(os.path.join(processed_data_dir, "add_lags_rollings.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "add_lags_rollings.pkl"))

    # add lags
    SHIFT_DAY = 28
    LAG_DAYS = [col for col in range(SHIFT_DAY, SHIFT_DAY + 15)]

    grid_df = grid_df.assign(
        **{
            "{}_lag_{}".format(col, l): grid_df.groupby(["id"])[col].transform(lambda x: x.shift(l))
            for l in LAG_DAYS
            for col in [TARGET]
        }
    )

    for col in list(grid_df):
        if "lag" in col:
            grid_df[col] = grid_df[col].astype(np.float16)

    # add rollings
    for i in [7, 14, 30, 60, 180]:
        grid_df["rolling_mean_" + str(i)] = (
            grid_df.groupby(["id"])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
        )
        grid_df["rolling_std_" + str(i)] = (
            grid_df.groupby(["id"])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)
        )

    # sliding window
    for d_shift in [1, 7, 14]:
        for d_window in [7, 14, 30, 60]:
            col_name = "rolling_mean_tmp_" + str(d_shift) + "_" + str(d_window)
            grid_df[col_name] = (
                grid_df.groupby(["id"])[TARGET]
                .transform(lambda x: x.shift(SHIFT_DAY + d_shift).rolling(d_window).mean())
                .astype(np.float16)
            )

    grid_df.to_pickle(os.path.join(processed_data_dir, "add_lags_rollings.pkl"))
    return grid_df


def add_mean_enc(grid_df):
    if os.path.exists(os.path.join(processed_data_dir, "add_mean_enc.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "add_mean_enc.pkl"))

    sales_df = dco(grid_df["sales"])
    grid_df["sales"][grid_df["d"] > (1941 - 28)] = np.nan

    icols = [
        ["state_id"],
        ["store_id"],
        ["cat_id"],
        ["dept_id"],
        ["state_id", "cat_id"],
        ["state_id", "dept_id"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
        ["item_id"],
        ["item_id", "state_id"],
        ["item_id", "store_id"],
    ]

    for col in icols:
        col_name = "_" + "_".join(col) + "_"
        grid_df["enc" + col_name + "mean"] = grid_df.groupby(col)["sales"].transform("mean").astype(np.float16)
        grid_df["enc" + col_name + "std"] = grid_df.groupby(col)["sales"].transform("std").astype(np.float16)

    grid_df["sales"] = sales_df

    grid_df.to_pickle(os.path.join(processed_data_dir, "add_mean_enc.pkl"))
    return grid_df


def add_snap(grid_df):
    if os.path.exists(os.path.join(processed_data_dir, "all_data_df.pkl")):
        return pd.read_pickle(os.path.join(processed_data_dir, "all_data_df.pkl"))

    mask_CA = grid_df["state_id"] == "CA"
    mask_WI = grid_df["state_id"] == "WI"
    mask_TX = grid_df["state_id"] == "TX"

    grid_df["snap"] = grid_df["snap_CA"]
    grid_df.loc[mask_WI, "snap"] = grid_df["snap_WI"]
    grid_df.loc[mask_TX, "snap"] = grid_df["snap_TX"]

    grid_df.to_pickle(os.path.join(processed_data_dir, "all_data_df.pkl"))
    return grid_df


def preprocessing_m5():
    train_df = pd.read_csv(os.path.join(raw_data_dir, "sales_train_evaluation.csv"))
    prices_df = pd.read_csv(os.path.join(raw_data_dir, "sell_prices.csv"))
    calendar_df = pd.read_csv(os.path.join(raw_data_dir, "calendar.csv"))

    grid_df = melt_raw_data(train_df)
    print(f"df: ({grid_df.shape[0]}, {grid_df.shape[1]})  Melting raw data down!")

    grid_df = add_release_week(grid_df, prices_df, calendar_df)
    print(f"df: ({grid_df.shape[0]}, {grid_df.shape[1]})  Adding release week down!")

    grid_df = add_prices(grid_df, prices_df, calendar_df)
    print(f"df: ({grid_df.shape[0]}, {grid_df.shape[1]})  Adding prices down!")

    grid_df = add_date(grid_df, calendar_df)
    print(f"df: ({grid_df.shape[0]}, {grid_df.shape[1]})  Adding date down!")

    grid_df = add_lags_rollings(grid_df)
    print(f"df: ({grid_df.shape[0]}, {grid_df.shape[1]})  Adding lags and rollings down!")

    grid_df = add_mean_enc(grid_df)
    print(f"df: ({grid_df.shape[0]}, {grid_df.shape[1]})  Adding mean encoding down!")

    grid_df = pd.read_pickle(os.path.join(processed_data_dir, "add_mean_enc.pkl"))

    grid_df = add_snap(grid_df)
    print("Save the data down!")


# ==================== split dataset ====================
def label_encode(df, columns):
    le = LabelEncoder()
    data_list = []

    for column in columns:
        data_list += df[column].drop_duplicates().values.tolist()
    le.fit(data_list)

    for column in columns:
        df[column] = le.transform(df[column].values.tolist())

    return df


def reorganize_data(grid_df):
    grid_df["snap"] = grid_df["snap"].astype("int8")
    columns_list = [
        ["item_id"],
        ["dept_id"],
        ["cat_id"],
        ["event_name_1", "event_name_2"],
        ["event_type_1", "event_type_2"],
    ]

    for columns in columns_list:
        grid_df[columns] = label_encode(grid_df[columns], columns)

    return reduce_mem_usage(grid_df)


def split_data(df, store, fill_flag=False):
    for cat in category_list:
        df[cat] = df[cat].astype("category")

    if fill_flag:
        df = reduce_mem_usage(df, float16_flag=False)
        cols = df.isnull().any()
        idx = list(cols[cols.values].index)

        df[idx] = df.groupby("item_id", sort=False)[idx].apply(lambda x: x.ffill().bfill())
        df[idx] = df[idx].fillna(df[idx].mean())

        mms = MinMaxScaler()
        df[features_columns] = mms.fit_transform(df[features_columns])

        df = reduce_mem_usage(df)

    train_df = df[df["d"] <= END_TRAIN]
    val_df = df[df["d"] > END_TRAIN]

    train_df = train_df[features_columns + label_column]
    val_df = val_df[features_columns + label_column]
    print(train_df.shape, val_df.shape)

    suffix = f"_fill" if fill_flag else ""
    train_df.to_pickle(os.path.join(processed_data_dir, f"train_{store}{suffix}.pkl"))
    val_df.to_pickle(os.path.join(processed_data_dir, f"val_{store}{suffix}.pkl"))


def split_m5():
    grid_df = pd.read_pickle(os.path.join(processed_data_dir, "all_data_df.pkl"))

    if os.path.exists(os.path.join(processed_data_dir, "label_encode.pkl")):
        grid_df = pd.read_pickle(os.path.join(processed_data_dir, "label_encode.pkl"))
    else:
        grid_df = reorganize_data(grid_df)
        grid_df.to_pickle(os.path.join(processed_data_dir, "label_encode.pkl"))

    for store in store_list:
        # split_data(grid_df[grid_df["store_id"] == store], store)
        split_data(grid_df[grid_df["store_id"] == store], store, True)


def regenerate_data():
    preprocessing_m5()
    split_m5()
