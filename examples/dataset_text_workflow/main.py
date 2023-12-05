import os
import fire
import pickle
import time
import zipfile
from shutil import copyfile, rmtree

import numpy as np

import learnware.specification as specification
from get_data import get_data
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser
from utils import generate_uploader, generate_user, TextDataLoader, train, eval_prediction

logger = get_module_logger("text_test", level="INFO")
origin_data_root = "./data/origin_data"
processed_data_root = "./data/processed_data"
tmp_dir = "./data/tmp"
learnware_pool_dir = "./data/learnware_pool"
dataset = "ae"  # argumentative essays
n_uploaders = 7
n_users = 7
n_classes = 3
data_root = os.path.join(origin_data_root, dataset)
data_save_root = os.path.join(processed_data_root, dataset)
user_save_root = os.path.join(data_save_root, "user")
uploader_save_root = os.path.join(data_save_root, "uploader")
model_save_root = os.path.join(data_save_root, "uploader_model")
os.makedirs(data_root, exist_ok=True)
os.makedirs(user_save_root, exist_ok=True)
os.makedirs(uploader_save_root, exist_ok=True)
os.makedirs(model_save_root, exist_ok=True)

output_description = {
    "Dimension": 1,
    "Description": {
        "0": "classify as 0(ineffective), 1(effective), or 2(adequate).",
    },
}
semantic_specs = [
    {
        "Data": {"Values": ["Text"], "Type": "Class"},
        "Task": {"Values": ["Classification"], "Type": "Class"},
        "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
        "Scenario": {"Values": ["Education"], "Type": "Tag"},
        "Description": {"Values": "", "Type": "String"},
        "Name": {"Values": "learnware_1", "Type": "String"},
        "Output": output_description,
        "License": {"Values": ["MIT"], "Type": "Class"},
    }
]

user_semantic = {
    "Data": {"Values": ["Text"], "Type": "Class"},
    "Task": {"Values": ["Classification"], "Type": "Class"},
    "Library": {"Values": ["Scikit-learn"], "Type": "Class"},
    "Scenario": {"Values": ["Education"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
    "Output": output_description,
    "License": {"Values": ["MIT"], "Type": "Class"},
}


class TextDatasetWorkflow:
    def _init_text_dataset(self):
        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        X_train, y_train, X_test, y_test = get_data(data_root)

        generate_uploader(X_train, y_train, n_uploaders=n_uploaders, data_save_root=uploader_save_root)
        generate_user(X_test, y_test, n_users=n_users, data_save_root=user_save_root)

    def _prepare_model(self):
        dataloader = TextDataLoader(data_save_root, train=True)
        for i in range(n_uploaders):
            logger.info("Train on uploader: %d" % (i))
            X, y = dataloader.get_idx_data(i)
            vectorizer, lgbm = train(X, y, out_classes=n_classes)

            modelv_save_path = os.path.join(model_save_root, "uploader_v_%d.pth" % (i))
            modell_save_path = os.path.join(model_save_root, "uploader_l_%d.pth" % (i))

            with open(modelv_save_path, "wb") as f:
                pickle.dump(vectorizer, f)

            with open(modell_save_path, "wb") as f:
                pickle.dump(lgbm, f)

            logger.info("Model saved to '%s' and '%s'" % (modelv_save_path, modell_save_path))

    def _prepare_learnware(
        self, data_path, modelv_path, modell_path, init_file_path, yaml_path, env_file_path, save_root, zip_name
    ):
        os.makedirs(save_root, exist_ok=True)
        tmp_spec_path = os.path.join(save_root, "rkme.json")

        tmp_modelv_path = os.path.join(save_root, "modelv.pth")
        tmp_modell_path = os.path.join(save_root, "modell.pth")

        tmp_yaml_path = os.path.join(save_root, "learnware.yaml")
        tmp_init_path = os.path.join(save_root, "__init__.py")
        tmp_env_path = os.path.join(save_root, "requirements.txt")

        with open(data_path, "rb") as f:
            X = pickle.load(f)
        semantic_spec = semantic_specs[0]

        st = time.time()

        user_spec = specification.RKMETextSpecification()

        user_spec.generate_stat_spec_from_data(X=X)
        ed = time.time()
        logger.info("Stat spec generated in %.3f s" % (ed - st))
        user_spec.save(tmp_spec_path)

        copyfile(modelv_path, tmp_modelv_path)
        copyfile(modell_path, tmp_modell_path)

        copyfile(yaml_path, tmp_yaml_path)
        copyfile(init_file_path, tmp_init_path)
        copyfile(env_file_path, tmp_env_path)
        zip_file_name = os.path.join(learnware_pool_dir, "%s.zip" % (zip_name))
        with zipfile.ZipFile(zip_file_name, "w", compression=zipfile.ZIP_DEFLATED) as zip_obj:
            zip_obj.write(tmp_spec_path, "rkme.json")

            zip_obj.write(tmp_modelv_path, "modelv.pth")
            zip_obj.write(tmp_modell_path, "modell.pth")

            zip_obj.write(tmp_yaml_path, "learnware.yaml")
            zip_obj.write(tmp_init_path, "__init__.py")
            zip_obj.write(tmp_env_path, "requirements.txt")
        rmtree(save_root)
        logger.info("New Learnware Saved to %s" % (zip_file_name))
        return zip_file_name

    def prepare_market(self, regenerate_flag=False):
        if regenerate_flag:
            self._init_text_dataset()
        text_market = instantiate_learnware_market(market_id="ae", rebuild=True)
        try:
            rmtree(learnware_pool_dir)
        except:
            pass
        os.makedirs(learnware_pool_dir, exist_ok=True)
        for i in range(n_uploaders):
            data_path = os.path.join(uploader_save_root, "uploader_%d_X.pkl" % (i))

            modelv_path = os.path.join(model_save_root, "uploader_v_%d.pth" % (i))
            modell_path = os.path.join(model_save_root, "uploader_l_%d.pth" % (i))

            init_file_path = "./example_files/example_init.py"
            yaml_file_path = "./example_files/example_yaml.yaml"
            env_file_path = "./example_files/requirements.txt"
            new_learnware_path = self._prepare_learnware(
                data_path,
                modelv_path,
                modell_path,
                init_file_path,
                yaml_file_path,
                env_file_path,
                tmp_dir,
                "%s_%d" % (dataset, i),
            )
            semantic_spec = semantic_specs[0]
            semantic_spec["Name"]["Values"] = "learnware_%d" % (i)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (i)
            text_market.add_learnware(new_learnware_path, semantic_spec)

        logger.info("Total Item: %d" % (len(text_market)))

    def test(self, regenerate_flag=False):
        self.prepare_market(regenerate_flag)
        text_market = instantiate_learnware_market(market_id="ae")
        print("Total Item: %d" % len(text_market))

        select_list = []
        avg_list = []
        improve_list = []
        job_selector_score_list = []
        ensemble_score_list = []
        pruning_score_list = []
        for i in range(n_users):
            user_data_path = os.path.join(user_save_root, "user_%d_X.pkl" % (i))
            user_label_path = os.path.join(user_save_root, "user_%d_y.pkl" % (i))
            with open(user_data_path, "rb") as f:
                user_data = pickle.load(f)
            with open(user_label_path, "rb") as f:
                user_label = pickle.load(f)

            user_stat_spec = specification.RKMETextSpecification()
            user_stat_spec.generate_stat_spec_from_data(X=user_data)
            user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETextSpecification": user_stat_spec})
            logger.info("Searching Market for user: %d" % (i))

            search_result = text_market.search_learnware(user_info)
            single_result = search_result.get_single_results()
            multiple_result = search_result.get_multiple_results()

            print(f"search result of user{i}:")
            print(
                f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
            )

            l = len(single_result)
            acc_list = []
            for idx in range(l):
                learnware = single_result[idx].learnware
                score = single_result[idx].score
                pred_y = learnware.predict(user_data)
                acc = eval_prediction(pred_y, user_label)
                acc_list.append(acc)
            print(
                f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, acc: {acc_list[0]}"
            )

            if len(multiple_result) > 0:
                mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                print(f"mixture_score: {multiple_result[0].score}, mixture_learnware: {mixture_id}")
                mixture_learnware_list = multiple_result[0].learnwares
            else:
                mixture_learnware_list = [single_result[0].learnware]

            # test reuse (job selector)
            reuse_baseline = JobSelectorReuser(learnware_list=mixture_learnware_list, herding_num=100)
            reuse_predict = reuse_baseline.predict(user_data=user_data)
            reuse_score = eval_prediction(reuse_predict, user_label)
            job_selector_score_list.append(reuse_score)
            print(f"mixture reuse loss(job selector): {reuse_score}")

            # test reuse (ensemble)
            reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote_by_label")
            ensemble_predict_y = reuse_ensemble.predict(user_data=user_data)
            ensemble_score = eval_prediction(ensemble_predict_y, user_label)
            ensemble_score_list.append(ensemble_score)
            print(f"mixture reuse accuracy (ensemble): {ensemble_score}")

            # test reuse (ensemblePruning)
            reuse_pruning = EnsemblePruningReuser(learnware_list=mixture_learnware_list)
            pruning_predict_y = reuse_pruning.predict(user_data=user_data)
            pruning_score = eval_prediction(pruning_predict_y, user_label)
            pruning_score_list.append(pruning_score)
            print(f"mixture reuse accuracy (ensemble Pruning): {pruning_score}\n")

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
            "Averaging Ensemble Reuse Performance: %.3f +/- %.3f"
            % (np.mean(ensemble_score_list), np.std(ensemble_score_list))
        )
        logger.info(
            "Selective Ensemble Reuse Performance: %.3f +/- %.3f"
            % (np.mean(pruning_score_list), np.std(pruning_score_list))
        )


if __name__ == "__main__":
    fire.Fire(TextDatasetWorkflow)
