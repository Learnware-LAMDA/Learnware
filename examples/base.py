import os
import joblib
import zipfile
from shutil import copyfile, rmtree

import json
from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market
from multiprocessing import Pool

from benchmarks import DataLoader
from config import *
from methods import *
from utils import process_single_aug

logger = get_module_logger("TableWorkflow", level="INFO")


class TableWorkflow:
    def __init__(self, learnware_market):
        self.learnware_market = learnware_market

        self.root_path = os.path.abspath(os.path.join(__file__, ".."))
        self.learnware_pool_path = os.path.join(self.root_path, "data/learnware_pool")
        self.learnware_zip_pool_path = os.path.join(self.root_path, "data/zips")
        self.example_learnware_path = os.path.join(self.root_path, "data/example_files")
        self.model_save_path = os.path.join(self.root_path, "data/uploader_models")
        self.result_path = os.path.join(self.root_path, "results")

        os.makedirs(self.learnware_pool_path, exist_ok=True)
        os.makedirs(self.learnware_zip_pool_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    def _init_dataset(self):
        self._prepare_data()
        self._prepare_model()
    
    @staticmethod
    def _limited_data(method, test_info, loss_func):
        all_scores = []
        for subset in test_info["train_subsets"]:
            subset_scores = []
            for sample in subset:
                x_train, y_train = sample["x_train"], sample["y_train"]
                model = method(x_train, y_train, test_info)
                subset_scores.append(loss_func(model.predict(test_info["test_x"]), test_info["test_y"]))
            all_scores.append(np.mean(subset_scores))
        return all_scores
    
    # @staticmethod
    # def _limited_data_single_learnware(method, test_info, learnware):
    #     test_info['single_learnware'] = learnware
    #     return TableWorkflow._limited_data(method, test_info)
    
    def test_method(self, test_info, recorders, loss_func=loss_func_rmse):
        method_name_full = test_info["method_name"]
        method_name = method_name_full if method_name_full == "user_model" else "_".join(method_name_full.split("_")[1:])
        user, idx = test_info["user"], test_info["idx"]
        recorder = recorders[method_name_full]
        
        save_root_path = os.path.join(self.curves_result_path, f"{user}/{user}_{idx}")
        os.makedirs(save_root_path, exist_ok=True)
        save_path = os.path.join(save_root_path, f"{method_name}.json")
        
        if method_name == "single_aug":
            if test_info["force"] or recorder.should_test_method(user, idx, save_path):
                # with Pool() as pool:
                #     learnware_results = pool.starmap(
                #         self._limited_data_single_learnware,
                #         [(test_methods[method_name], test_info, learnware) for learnware in test_info['learnwares']]
                #     )
                # for scores in learnware_results:
                #     recorders[method_name].record(user, idx, scores)
                
                for learnware in test_info['learnwares']:
                    test_info['single_learnware'] = learnware
                    scores = self._limited_data(test_methods[method_name_full], test_info, loss_func)
                    recorder.record(user, idx, scores)

                process_single_aug(user, idx, scores, recorders, save_root_path)
                recorder.save(save_path)
                logger.info(f"Method {method_name} on {user}_{idx} finished")
            else:
                process_single_aug(user, idx, recorder.data[user][str(idx)], recorders, save_root_path)  
                logger.info(f"Method {method_name} on {user}_{idx} already exists")
        else:
            if test_info["force"] or recorder.should_test_method(user, idx, save_path):
                scores = self._limited_data(test_methods[method_name_full], test_info, loss_func)
                recorder.record(user, idx, scores)
                recorder.save(save_path)
                logger.info(f"Method {method_name} on {user}_{idx} finished")
            else:
                logger.info(f"Method {method_name} on {user}_{idx} already exists")
    
    def prepare_market(self, name, market_id, regenerate_flag=False):
        if regenerate_flag:
            self._init_dataset()
        market = instantiate_learnware_market(name=name, market_id=market_id, rebuild=True)
        client = LearnwareClient()

        full_descriptions_dir = os.path.join("./data/full_descriptions.json")
        with open(full_descriptions_dir, "rb") as f:
            full_descriptions = json.load(f)

        for uploader in self.learnware_market:
            data_loader = DataLoader(uploader)
            idx_list = data_loader.get_shop_ids()
            for i, idx in enumerate(idx_list):
                feature_descriptions = data_loader.get_raw_data(idx)[-1]
                feature_dim = len(feature_descriptions)
                feature_descriptions_dict = {str(i): feature_descriptions[i] for i in range(feature_dim)}
                input_description = {"Dimension": feature_dim, "Description": feature_descriptions_dict}

                name_and_description = full_descriptions[uploader][i]
                semantic_spec = client.create_semantic_specification(
                    name=name_and_description["name"],
                    description=name_and_description["description"],
                    data_type="Table",
                    task_type="Regression",
                    library_type="Others",
                    license=["MIT"],
                    scenarios=["Business"],
                    input_description=input_description,
                    output_description=output_description,
                )

                learnware_zip_path = self._prepare_learnware(data_loader, idx)
                market.add_learnware(learnware_zip_path, semantic_spec)

        # if use pretrained market mapping
        if name == "hetero":
            learnware_ids = market.get_learnware_ids()
            market.learnware_organizer._update_learware_hetero_spec(learnware_ids)

        logger.info("Total Item: %d" % (len(market)))

    def _prepare_data(self):
        for uploader in self.learnware_market:
            data_loader = DataLoader(uploader)
            data_loader.regenerate_raw_data()

    def _prepare_model(self, use_exist=True):
        self.learnware_num = 0
        for uploader in self.learnware_market:
            data_loader = DataLoader(uploader)
            idx_list = data_loader.get_shop_ids()
            self.learnware_num += len(idx_list)
            for idx in idx_list:
                logger.info(f"Train on uploader: {uploader}_{idx}")
                idx_model_save_path = os.path.join(self.model_save_path, f"{uploader}_{idx}.out")
                if not use_exist:
                    x_train, y_train, x_val, y_val, _ = data_loader.get_raw_data(idx)
                    data_loader.train_a_model(x_train, y_train, x_val, y_val, save_dir=idx_model_save_path)
                else:
                    uploader_dataset = uploader.split("_")[0]
                    model = data_loader.get_model(idx)
                    if uploader_dataset == "corporacion":
                        model.save_model(idx_model_save_path)
                    elif uploader_dataset == "pfs":
                        joblib.dump(model, idx_model_save_path)
                    else:
                        logger.error(f"Not supported dataset type {uploader_dataset}")

                logger.info(f"Model saved to {idx_model_save_path}")

    def _prepare_learnware(self, data_loader, idx):
        zip_path = os.path.join(self.learnware_zip_pool_path, f"{data_loader.dataset}_{idx}")
        dir_path = os.path.join(self.learnware_pool_path, f"{data_loader.dataset}_{idx}")
        model_path = os.path.join(self.model_save_path, f"{data_loader.dataset}_{idx}.out")
        os.makedirs(dir_path, exist_ok=True)

        stat_spec, _ = data_loader.get_rkme(idx)
        init_file = os.path.join(dir_path, "__init__.py")
        yaml_file = os.path.join(dir_path, "learnware.yaml")
        env_file = os.path.join(dir_path, "environment.yaml")
        model_file = os.path.join(dir_path, "model.out")

        stat_spec.save(os.path.join(dir_path, "rkme.json"))
        copyfile(os.path.join(self.example_learnware_path, f"{data_loader.dataset}/__init__.py"), init_file)
        copyfile(os.path.join(self.example_learnware_path, f"{data_loader.dataset}/learnware.yaml"), yaml_file)
        copyfile(os.path.join(self.example_learnware_path, "environment.yaml"), env_file)
        copyfile(model_path, model_file)

        zip_file = zip_path + ".zip"
        with zipfile.ZipFile(zip_file, "w") as zip_obj:
            for foldername, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_info = zipfile.ZipInfo(filename)
                    zip_info.compress_type = zipfile.ZIP_STORED
                    with open(file_path, "rb") as file:
                        zip_obj.writestr(zip_info, file.read())

        rmtree(dir_path)  # rm -r dir_path
        return zip_file

   