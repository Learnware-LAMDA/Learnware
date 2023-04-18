import numpy as np
import torch
import get_data
import os
import random
from utils import generate_uploader, generate_user, ImageDataLoader, train

from learnware.market import EasyMarket, BaseUserInfo
from learnware.market import database_ops
from learnware.learnware import Learnware
import learnware.specification as specification

from shutil import copyfile, rmtree
import zipfile

origin_data_root = "./data/origin_data"
processed_data_root = "./data/processed_data"
tmp_dir = "./data/tmp"
learnware_pool_dir = "./data/learnware_pool"
dataset = "cifar10"
n_uploaders = 50
n_users = 10
n_classes = 10
data_root = os.path.join(origin_data_root, dataset)
data_save_root = os.path.join(processed_data_root, dataset)
user_save_root = os.path.join(data_save_root, "user")
uploader_save_root = os.path.join(data_save_root, "uploader")
model_save_root = os.path.join(data_save_root, "uploader_model")
os.makedirs(data_root, exist_ok=True)
os.makedirs(user_save_root, exist_ok=True)
os.makedirs(uploader_save_root, exist_ok=True)
os.makedirs(model_save_root, exist_ok=True)


semantic_specs = [
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {
            "Values": ["Classification"],
            "Type": "Class",
        },
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Nature"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "learnware_1", "Type": "Name"},
    },
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {
            "Values": ["Classification"],
            "Type": "Class",
        },
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business", "Nature"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "learnware_2", "Type": "Name"},
    },
    {
        "Data": {"Values": ["Tabular"], "Type": "Class"},
        "Task": {
            "Values": ["Classification"],
            "Type": "Class",
        },
        "Device": {"Values": ["GPU"], "Type": "Tag"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "Description"},
        "Name": {"Values": "learnware_3", "Type": "Name"},
    },
]

user_senmantic = {
    "Data": {"Values": ["Tabular"], "Type": "Class"},
    "Task": {
        "Values": ["Classification"],
        "Type": "Class",
    },
    "Device": {"Values": ["GPU"], "Type": "Tag"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "Description"},
    "Name": {"Values": "", "Type": "Name"},
}


def prepare_data():
    if dataset == "cifar10":
        X_train, y_train, X_test, y_test = get_data.get_cifar10(data_root)
    elif dataset == "mnist":
        X_train, y_train, X_test, y_test = get_data.get_mnist(data_root)
    else:
        return
    generate_uploader(X_train, y_train, n_uploaders=n_uploaders, data_save_root=uploader_save_root)
    generate_user(X_test, y_test, n_users=n_users, data_save_root=user_save_root)


def prepare_model():
    dataloader = ImageDataLoader(data_save_root, train=True)
    for i in range(n_uploaders):
        print("Train on uploader: %d" % (i))
        X, y = dataloader.get_idx_data(i)
        model = train(X, y, out_classes=n_classes)
        model_save_path = os.path.join(model_save_root, "uploader_%d.pth" % (i))
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to '%s'" % (model_save_path))


def prepare_learnware(data_path, model_path, init_file_path, yaml_path):
    X = np.load(data_path)
    user_spec = specification.utils.generate_rkme_spec(X=X, gamma=0.1, cuda_idx=0)
    print(user_spec.shape)


def prepare_market():
    image_market = EasyMarket(rebuild=True)
    os.makedirs(learnware_pool_dir)
    for i in range(n_uploaders):
        data_path = os.path.join(uploader_save_root, "uploader_%d_X.npy" % (i))
        model_path = os.path.join(model_save_root, "uploader_%d.pth" % (i))
        init_file_path = "./example_init.py"
        yaml_file_path = "./example_yaml.yaml"
        prepare_learnware(data_path, model_path, init_file_path, yaml_file_path)


if __name__ == "__main__":
    # prepare_data()
    # prepare_model()
    prepare_market()
