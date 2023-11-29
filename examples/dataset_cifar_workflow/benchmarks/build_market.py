import copy
import os
import zipfile
from shutil import copyfile, rmtree

import learnware
import numpy as np
import tqdm
import yaml
from learnware import specification
from learnware.market import EasyMarket

from preprocess.dataloader import ImageDataLoader

user_semantic = {
    "Data": {"Values": ["Image"], "Type": "Class"},
    "Task": {
        "Values": ["Classification"],
        "Type": "Class",
    },
    "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
    "Scenario": {"Values": [], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
    "Output": {"Values": "", "Dimension": 0}
}


def build_from_preprocessed(args, regenerate=True):
    zip_path_list = []
    data_root = os.path.join(args.data_root, 'learnware_market_data', "{}_{:d}".format(args.data, args.data_id))
    dataloader = ImageDataLoader(data_root, args.n_uploaders, train=True)

    market_root = args.market_root
    for i, (train_X, train_y, val_X, val_y) in tqdm.tqdm(enumerate(dataloader), total=args.n_uploaders):
        dir_path = os.path.join(market_root, args.data, "{}_{:d}".format(args.spec, args.id), "learnware_{:d}".format(i))
        os.makedirs(dir_path, exist_ok=True)

        if not regenerate:
            zip_path_list.append(dir_path + ".zip")
            continue

        # print("Preparing Learnware {:d} with {:s} specification".format(i, args.spec))
        # Copy Model File
        model_file = os.path.join(dir_path, "model.pth")
        copyfile(os.path.join(data_root, "models", "uploader_{:d}.pth".format(i)),
                 model_file)

        # Make Specification
        if args.spec == "rbf":
            spec = specification.utils.generate_rkme_spec(X=train_X, reduced_set_size=args.K, gamma=0.1, cuda_idx=args.cuda_idx)
        elif args.spec == "ntk":
            spec = learnware.specification.RKMEImageStatSpecification(rkme_id=i, **args.__dict__)
            spec.generate_stat_spec_from_data(val_X, K=args.K, steps=args.ntk_steps, reduce=True, whitening=False)
        else:
            raise NotImplementedError("Not Support", args.spec)
        spec.save(os.path.join(dir_path, "spec.json"))

        # Copy __init__.py and learnware_yaml
        init_file = os.path.join(dir_path, "__init__.py")
        yaml_file = os.path.join(dir_path, "learnware.yaml")
        copyfile(
            os.path.join(market_root, "learnware_example",
                         "conv.py"), init_file
        )  # cp conv.py init_file

        with open(os.path.join(market_root, "learnware_example",
                              "{}.yaml".format(args.spec)), "r") as yaml_templet,\
            open(yaml_file, "w") as yaml_target:

            yaml_content = yaml.load(yaml_templet, Loader=yaml.FullLoader)

            yaml_content["model"]["kwargs"]["device"] = str(choose_device(args.cuda_idx))
            yaml_content["model"]["kwargs"]["input_channel"] = train_X.shape[1]
            if args.spec == "ntk":
                yaml_content["stat_specifications"][0]["kwargs"] = copy.deepcopy(args.__dict__)

            yaml.dump(yaml_content, yaml_target)


        zip_file = dir_path + ".zip"
        # zip -q -r -j zip_file dir_path
        with zipfile.ZipFile(zip_file, "w") as zip_obj:
            for foldername, subfolders, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_info = zipfile.ZipInfo(filename)
                    zip_info.compress_type = zipfile.ZIP_STORED
                    with open(file_path, "rb") as file:
                        zip_obj.writestr(zip_info, file.read())

        rmtree(dir_path)  # rm -r dir_path
        zip_path_list.append(zip_file)

    return zip_path_list

def upload_to_easy_market(args, zip_path_list, market_id=None):
    learnware.init()
    np.random.seed(2023)
    market_id = market_id if market_id else "NTK-RF-{:d}".format(args.id)
    market = DummyMarket(market_id=market_id, rebuild=True)

    for idx, zip_path in enumerate(zip_path_list):
        semantic_spec = copy.deepcopy(user_semantic)
        semantic_spec["Name"]["Values"] = "learnware_{:d}".format(idx)
        semantic_spec["Description"]["Values"] = "test_learnware_number_{:d}".format(idx)
        semantic_spec["Scenario"]["Values"] = [args.data]
        semantic_spec["Output"]['Dimension'] = 10
        market.add_learnware(zip_path, semantic_spec)

    logger = get_custom_logger()
    logger.debug("Total Item: {:d}".format(len(market)))

    return market