import numpy as np
import torch
from get_data import *
import os
import random
from utils import generate_uploader, generate_user, ImageDataLoader, train, eval_prediction
from learnware.learnware import Learnware, JobSelectorReuser, AveragingReuser
import time

from learnware.market import EasyMarket, BaseUserInfo
from learnware.market import database_ops
from learnware.learnware import Learnware
import learnware.specification as specification
from learnware.logger import get_module_logger

from shutil import copyfile, rmtree
import zipfile

logger = get_module_logger("image_test", level="INFO")
origin_data_root = "./data/origin_data"
processed_data_root = "./data/processed_data"
tmp_dir = "./data/tmp"
learnware_pool_dir = "./data/learnware_pool"
dataset = "cifar10"
n_uploaders = 50
n_users = 20
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
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Library": {"Values": ["Pytorch"], "Type": "Class"},
        "Scenario": {"Values": ["Business"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "learnware_1", "Type": "String"},
    }
]

user_semantic = {
    "Data": {"Values": ["Tabular"], "Type": "Class"},
    "Task": {"Values": ["Classification"], "Type": "Class"},
    "Library": {"Values": ["Pytorch"], "Type": "Class"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
}


def prepare_data():
    if dataset == "cifar10":
        X_train, y_train, X_test, y_test = get_cifar10(data_root)
    elif dataset == "mnist":
        X_train, y_train, X_test, y_test = get_mnist(data_root)
    else:
        return
    generate_uploader(X_train, y_train, n_uploaders=n_uploaders, data_save_root=uploader_save_root)
    generate_user(X_test, y_test, n_users=n_users, data_save_root=user_save_root)


def prepare_model():
    dataloader = ImageDataLoader(data_save_root, train=True)
    for i in range(n_uploaders):
        logger.info("Train on uploader: %d" % (i))
        X, y = dataloader.get_idx_data(i)
        model = train(X, y, out_classes=n_classes)
        model_save_path = os.path.join(model_save_root, "uploader_%d.pth" % (i))
        torch.save(model.state_dict(), model_save_path)
        logger.info("Model saved to '%s'" % (model_save_path))


def prepare_learnware(data_path, model_path, init_file_path, yaml_path, save_root, zip_name):
    os.makedirs(save_root, exist_ok=True)
    tmp_spec_path = os.path.join(save_root, "rkme.json")
    tmp_model_path = os.path.join(save_root, "conv_model.pth")
    tmp_yaml_path = os.path.join(save_root, "learnware.yaml")
    tmp_init_path = os.path.join(save_root, "__init__.py")
    tmp_model_file_path = os.path.join(save_root, "model.py")
    mmodel_file_path = "./example_files/model.py"
    X = np.load(data_path)
    st = time.time()
    user_spec = specification.utils.generate_rkme_spec(X=X, gamma=0.1, cuda_idx=0)
    ed = time.time()
    logger.info("Stat spec generated in %.3f s" % (ed - st))
    user_spec.save(tmp_spec_path)
    copyfile(model_path, tmp_model_path)
    copyfile(yaml_path, tmp_yaml_path)
    copyfile(init_file_path, tmp_init_path)
    copyfile(mmodel_file_path, tmp_model_file_path)
    zip_file_name = os.path.join(learnware_pool_dir, "%s.zip" % (zip_name))
    with zipfile.ZipFile(zip_file_name, "w", compression=zipfile.ZIP_DEFLATED) as zip_obj:
        zip_obj.write(tmp_spec_path, "rkme.json")
        zip_obj.write(tmp_model_path, "conv_model.pth")
        zip_obj.write(tmp_yaml_path, "learnware.yaml")
        zip_obj.write(tmp_init_path, "__init__.py")
        zip_obj.write(tmp_model_file_path, "model.py")
    rmtree(save_root)
    logger.info("New Learnware Saved to %s" % (zip_file_name))
    return zip_file_name


def prepare_market():
    image_market = EasyMarket(market_id="cifar10", rebuild=True)
    try:
        rmtree(learnware_pool_dir)
    except:
        pass
    os.makedirs(learnware_pool_dir, exist_ok=True)
    for i in range(n_uploaders):
        data_path = os.path.join(uploader_save_root, "uploader_%d_X.npy" % (i))
        model_path = os.path.join(model_save_root, "uploader_%d.pth" % (i))
        init_file_path = "./example_files/example_init.py"
        yaml_file_path = "./example_files/example_yaml.yaml"
        new_learnware_path = prepare_learnware(
            data_path, model_path, init_file_path, yaml_file_path, tmp_dir, "%s_%d" % (dataset, i)
        )
        semantic_spec = semantic_specs[0]
        semantic_spec["Name"]["Values"] = "learnware_%d" % (i)
        semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (i)
        image_market.add_learnware(new_learnware_path, semantic_spec)

    logger.info("Total Item: %d" % (len(image_market)))
    curr_inds = image_market._get_ids()
    logger.info("Available ids: " + str(curr_inds))


def test_search(gamma=0.1, load_market=True):
    if load_market:
        image_market = EasyMarket(market_id="cifar10")
    else:
        prepare_market()
        image_market = EasyMarket(market_id="cifar10")
    logger.info("Number of items in the market: %d" % len(image_market))

    select_list = []
    avg_list = []
    improve_list = []
    job_selector_score_list = []
    ensemble_score_list = []
    for i in range(n_users):
        user_data_path = os.path.join(user_save_root, "user_%d_X.npy" % (i))
        user_label_path = os.path.join(user_save_root, "user_%d_y.npy" % (i))
        user_data = np.load(user_data_path)
        user_label = np.load(user_label_path)
        user_stat_spec = specification.utils.generate_rkme_spec(X=user_data, gamma=gamma, cuda_idx=0)
        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": user_stat_spec})
        logger.info("Searching Market for user: %d" % (i))
        sorted_score_list, single_learnware_list, mixture_score, mixture_learnware_list = image_market.search_learnware(
            user_info
        )
        l = len(sorted_score_list)
        acc_list = []
        for idx in range(l):
            learnware = single_learnware_list[idx]
            score = sorted_score_list[idx]
            pred_y = learnware.predict(user_data)
            acc = eval_prediction(pred_y, user_label)
            acc_list.append(acc)
            logger.info("search rank: %d, score: %.3f, learnware_id: %s, acc: %.3f" % (idx, score, learnware.id, acc))

        # test reuse (job selector)
        reuse_baseline = JobSelectorReuser(learnware_list=mixture_learnware_list, herding_num=100)
        reuse_predict = reuse_baseline.predict(user_data=user_data)
        reuse_score = eval_prediction(reuse_predict, user_label)
        job_selector_score_list.append(reuse_score)
        print(f"mixture reuse loss: {reuse_score}")

        # test reuse (ensemble)
        reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote")
        ensemble_predict_y = reuse_ensemble.predict(user_data=user_data)
        ensemble_score = eval_prediction(ensemble_predict_y, user_label)
        ensemble_score_list.append(ensemble_score)
        print(f"mixture reuse accuracy (ensemble): {ensemble_score}\n")

        select_list.append(acc_list[0])
        avg_list.append(np.mean(acc_list))
        improve_list.append((acc_list[0] - np.mean(acc_list)) / np.mean(acc_list))

    logger.info(
        "Accuracy of selected learnware: %.3f +/- %.3f, Average performance: %.3f +/- %.3f"
        % (np.mean(select_list), np.std(select_list), np.mean(avg_list), np.std(avg_list))
    )
    logger.info("Average performance improvement: %.3f" % (np.mean(improve_list)))
    logger.info(
        "Average Job Selector Reuse Performance: %.3f +/- %.3f"
        % (np.mean(job_selector_score_list), np.std(job_selector_score_list))
    )
    logger.info(
        "Ensemble Reuse Performance: %.3f +/- %.3f" % (np.mean(ensemble_score_list), np.std(ensemble_score_list))
    )


if __name__ == "__main__":
    prepare_data()
    prepare_model()
    test_search(load_market=False)