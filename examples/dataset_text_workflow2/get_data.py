import os

import pandas as pd


def get_data(data_root="./data"):
    dtrain = pd.read_csv(os.path.join(data_root, "train.csv"))
    dtest = pd.read_csv(os.path.join(data_root, "test.csv"))

    # returned X(DataFrame), y(Series)
    return (dtrain[['discourse_text', 'discourse_type']],
            dtrain["discourse_effectiveness"],
            dtest[['discourse_text', 'discourse_type']],
            dtest["discourse_effectiveness"])
