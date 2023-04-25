import os


ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "data"))
raw_data_dir = os.path.join(ROOT_PATH, "raw")
processed_data_dir = os.path.join(ROOT_PATH, "processed")
model_dir = os.path.join(ROOT_PATH, "models")
grid_dir = os.path.join(ROOT_PATH, "grid_sample")


TARGET = "sales"
START_TRAIN = 1
END_TRAIN = 1941 - 28


category_list = ["item_id", "dept_id", "cat_id", "event_name_1", "event_name_2", "event_type_1", "event_type_2"]
features_columns = [
    "item_id",
    "dept_id",
    "cat_id",
    "release",
    "sell_price",
    "price_max",
    "price_min",
    "price_std",
    "price_mean",
    "price_norm",
    "price_nunique",
    "item_nunique",
    "price_momentum",
    "price_momentum_m",
    "price_momentum_y",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "tm_d",
    "tm_w",
    "tm_m",
    "tm_y",
    "tm_wm",
    "tm_dw",
    "tm_w_end",
    "sales_lag_28",
    "sales_lag_29",
    "sales_lag_30",
    "sales_lag_31",
    "sales_lag_32",
    "sales_lag_33",
    "sales_lag_34",
    "sales_lag_35",
    "sales_lag_36",
    "sales_lag_37",
    "sales_lag_38",
    "sales_lag_39",
    "sales_lag_40",
    "sales_lag_41",
    "sales_lag_42",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_30",
    "rolling_std_30",
    "rolling_mean_60",
    "rolling_std_60",
    "rolling_mean_180",
    "rolling_std_180",
    "rolling_mean_tmp_1_7",
    "rolling_mean_tmp_1_14",
    "rolling_mean_tmp_1_30",
    "rolling_mean_tmp_1_60",
    "rolling_mean_tmp_7_7",
    "rolling_mean_tmp_7_14",
    "rolling_mean_tmp_7_30",
    "rolling_mean_tmp_7_60",
    "rolling_mean_tmp_14_7",
    "rolling_mean_tmp_14_14",
    "rolling_mean_tmp_14_30",
    "rolling_mean_tmp_14_60",
    # "enc_state_id_mean",
    # "enc_state_id_std",
    # "enc_store_id_mean",
    # "enc_store_id_std",
    "enc_cat_id_mean",
    "enc_cat_id_std",
    "enc_dept_id_mean",
    "enc_dept_id_std",
    "enc_state_id_cat_id_mean",
    "enc_state_id_cat_id_std",
    "enc_state_id_dept_id_mean",
    "enc_state_id_dept_id_std",
    "enc_store_id_cat_id_mean",
    "enc_store_id_cat_id_std",
    "enc_store_id_dept_id_mean",
    "enc_store_id_dept_id_std",
    "enc_item_id_mean",
    "enc_item_id_std",
    "enc_item_id_state_id_mean",
    "enc_item_id_state_id_std",
    "enc_item_id_store_id_mean",
    "enc_item_id_store_id_std",
]
label_column = ["sales"]


lgb_params_list = [
    [0.015, 224, 66],
    [0.01, 224, 50],
    [0.01, 300, 80],
    [0.015, 128, 50],
    [0.015, 300, 50],
    [0.01, 300, 66],
    [0.015, 300, 80],
    [0.15, 224, 80],
    [0.005, 300, 50],
    [0.015, 224, 50],
]


store_list = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
dataset_info = {
    "name": "M5",
    "range of date": "2011.01.29-2016.06.19",
    "description": "Walmart store, involves the unit sales of various products sold in the USA, organized in the form of grouped time series. More specifically, the dataset involves the unit sales of 3049 products, classified in 3 product categories (Hobbies, Foods, and Household).",
    "location": [
        "California, United States",
        "California, United States",
        "California, United States",
        "California, United States",
        "Texas, United States",
        "Texas, United States",
        "Texas, United States",
        "Wisconsin, United States",
        "Wisconsin, United States",
        "Wisconsin, United States",
    ],
}
