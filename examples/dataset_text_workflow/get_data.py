from torchtext.datasets import SST2


def get_sst2(data_root="./data"):
    train_datapipe = SST2(root="./data", split="train")

    X_train = [x[0] for x in train_datapipe]
    y_train = [x[1] for x in train_datapipe]

    dev_datapipe = SST2(root="./data", split="dev")

    X_test = [x[0] for x in dev_datapipe]
    y_test = [x[1] for x in dev_datapipe]
    return X_train, y_train, X_test, y_test
