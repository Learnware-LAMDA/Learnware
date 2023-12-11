import os
import fire
import pickle
import time
import zipfile
from shutil import copyfile, rmtree
import random

import numpy as np

import learnware.specification as specification
from get_data import get_data
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser, FeatureAugmentReuser
from utils import generate_uploader, generate_user, TextDataLoader, train, eval_prediction
from learnware.client import LearnwareClient, SemanticSpecificationKey
import matplotlib.pyplot as plt
from learnware.specification import generate_semantic_spec

# Login to Beiming system
client = LearnwareClient()

logger = get_module_logger("text_workflow", level="INFO")
origin_data_root = "./data/origin_data"
processed_data_root = "./data/processed_data"
tmp_dir = "./data/tmp"
learnware_pool_dir = "./data/learnware_pool"
dataset = "20newsgroups"

n_uploaders = 50 # max = 10 * n_samples
n_samples = 5
n_users = 10  # max = 10
n_classes = 20

n_labeled_list = [100, 200, 500, 1000, 2000, 4000]
repeated_list = [10, 10, 10, 3, 3, 3]

data_root = os.path.join(origin_data_root, dataset)
data_save_root = os.path.join(processed_data_root, dataset)
user_save_root = os.path.join(data_save_root, "user")
uploader_save_root = os.path.join(data_save_root, "uploader")
model_save_root = os.path.join(data_save_root, "uploader_model")
user_train_save_root = os.path.join(data_save_root, "user_train")

os.makedirs(data_root, exist_ok=True)
os.makedirs(user_save_root, exist_ok=True)
os.makedirs(uploader_save_root, exist_ok=True)
os.makedirs(model_save_root, exist_ok=True)
os.makedirs(user_train_save_root, exist_ok=True)

output_description = {
    "Dimension": 20,
    "Description": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
                    "7": "7", "8": "8", "9": "9", "10": "10", "11": "11", "12": "12", "13": "13",
                    "14": "14", "15": "15", "16": "16", "17": "17", "18": "18", "19": "19"}
}


semantic_spec = generate_semantic_spec(
    name="learnware_example",
    description="Just a example for text learnware",
    data_type="Text",
    task_type="Classification",
    library_type="Scikit-learn",
    scenarios=["Education"],
    license="MIT",
    input_description=None,
    output_description=output_description,
)

user_semantic = generate_semantic_spec(
    # name="learnware_example",
    description="Just a example for text learnware",
    data_type="Text",
    task_type="Classification",
    library_type="Scikit-learn",
    scenarios=["Education"],
    license="MIT",
    input_description=None,
    output_description=output_description,
)


class TextDatasetWorkflow:
    def _init_text_dataset(self):
        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        X_train, y_train, X_test, y_test = get_data(data_root)

        generate_uploader(X_train, y_train, n_uploaders=n_uploaders, n_samples=n_samples,
                          data_save_root=uploader_save_root)
        generate_user(X_test, y_test, n_users=n_users, data_save_root=user_save_root)

        generate_user(X_train, y_train, n_users=n_users, data_save_root=user_train_save_root)

    def _prepare_model(self):
        dataloader = TextDataLoader(data_save_root, train=True)
        for i in range(n_uploaders):
            logger.info("Train on uploader: %d" % (i))
            X, y = dataloader.get_idx_data(i)
            vectorizer, clf = train(X, y, out_classes=n_classes)

            modelv_save_path = os.path.join(model_save_root, "uploader_v_%d.pth" % (i))
            modell_save_path = os.path.join(model_save_root, "uploader_l_%d.pth" % (i))

            with open(modelv_save_path, "wb") as f:
                pickle.dump(vectorizer, f)

            with open(modell_save_path, "wb") as f:
                pickle.dump(clf, f)

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
        text_market = instantiate_learnware_market(market_id=dataset, rebuild=True)
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
            semantic_spec["Name"]["Values"] = "learnware_%d" % (i)
            semantic_spec["Description"]["Values"] = "test_learnware_number_%d" % (i)
            text_market.add_learnware(new_learnware_path, semantic_spec)

        logger.info("Total Item: %d" % (len(text_market)))

    def test_unlabeled(self, regenerate_flag=False):
        self.prepare_market(regenerate_flag)
        text_market = instantiate_learnware_market(market_id=dataset)
        print("Total Item: %d" % len(text_market))

        select_list = []
        avg_list = []
        best_list = []
        improve_list = []
        job_selector_score_list = []
        ensemble_score_list = []
        all_learnwares = text_market.get_learnwares()
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

            acc_list = []
            for idx in range(len(all_learnwares)):
                learnware = all_learnwares[idx]
                pred_y = learnware.predict(user_data)
                acc = eval_prediction(pred_y, user_label)
                acc_list.append(acc)

            learnware = single_result[0].learnware
            pred_y = learnware.predict(user_data)
            best_acc = eval_prediction(pred_y, user_label)
            best_list.append(np.max(acc_list))
            select_list.append(best_acc)
            avg_list.append(np.mean(acc_list))
            improve_list.append((best_acc - np.mean(acc_list)) / np.mean(acc_list))
            print(f"market mean accuracy: {np.mean(acc_list)}, market best accuracy: {np.max(acc_list)}")
            print(
                f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, acc: {best_acc}"
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
            print(f"mixture reuse accuracy (job selector): {reuse_score}")

            # test reuse (ensemble)
            # be careful with the ensemble mode
            reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote_by_label")
            ensemble_predict_y = reuse_ensemble.predict(user_data=user_data)
            ensemble_score = eval_prediction(ensemble_predict_y, user_label)
            ensemble_score_list.append(ensemble_score)
            print(f"mixture reuse accuracy (ensemble): {ensemble_score}")

            print("\n")

        logger.info(
            "Accuracy of selected learnware: %.3f +/- %.3f, Average performance: %.3f +/- %.3f, Best performance: %.3f +/- %.3f"
            % (
                np.mean(select_list),
                np.std(select_list),
                np.mean(avg_list),
                np.std(avg_list),
                np.mean(best_list),
                np.std(best_list),
            )
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

    def test_labeled(self, regenerate_flag=False, train_flag=True):
        if train_flag:
            self.prepare_market(regenerate_flag)
            text_market = instantiate_learnware_market(market_id=dataset)
            print("Total Item: %d" % len(text_market))

            os.makedirs("./figs", exist_ok=True)
            os.makedirs("./curves", exist_ok=True)

            for i in range(n_users):
                user_model_score_mat = []
                pruning_score_mat = []
                single_score_mat = []
                user_data_path = os.path.join(user_save_root, "user_%d_X.pkl" % (i))
                user_label_path = os.path.join(user_save_root, "user_%d_y.pkl" % (i))
                with open(user_data_path, "rb") as f:
                    test_x = pickle.load(f)
                with open(user_label_path, "rb") as f:
                    test_y = pickle.load(f)
                    test_y = np.array(test_y)

                train_data_path = os.path.join(user_train_save_root, "user_%d_X.pkl" % (i))
                train_label_path = os.path.join(user_train_save_root, "user_%d_y.pkl" % (i))
                with open(train_data_path, "rb") as f:
                    train_x = pickle.load(f)
                with open(train_label_path, "rb") as f:
                    train_y = pickle.load(f)
                    train_y = np.array(train_y)

                user_stat_spec = specification.RKMETextSpecification()
                user_stat_spec.generate_stat_spec_from_data(X=test_x)
                user_info = BaseUserInfo(
                    semantic_spec=user_semantic, stat_info={"RKMETextSpecification": user_stat_spec}
                )
                logger.info(f"Searching Market for user_{i}")

                search_result = text_market.search_learnware(user_info)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                learnware = single_result[0].learnware
                pred_y = learnware.predict(test_x)
                best_acc = eval_prediction(pred_y, test_y)

                print(f"search result of user_{i}:")
                print(
                    f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}, single model acc: {best_acc}"
                )

                if len(multiple_result) > 0:
                    mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                    print(f"mixture_score: {multiple_result[0].score}, mixture_learnware: {mixture_id}")
                    mixture_learnware_list = multiple_result[0].learnwares
                else:
                    mixture_learnware_list = [single_result[0].learnware]
                print(len(train_x))
                for n_label, repeated in zip(n_labeled_list, repeated_list):
                    user_model_score_list, reuse_pruning_score_list = [], []
                    if n_label > len(train_x):
                        n_label = len(train_x)
                    for _ in range(repeated):
                        # x_train, y_train = train_x[:n_label], train_y[:n_label]
                        x_train, y_train = zip(*random.sample(list(zip(train_x, train_y)), k=n_label))
                        x_train = list(x_train)
                        y_train = np.array(list(y_train))

                        modelv, modell = train(x_train, y_train, out_classes=n_classes)
                        user_model_predict_y = modell.predict(modelv.transform(test_x))
                        user_model_score = eval_prediction(user_model_predict_y, test_y)
                        user_model_score_list.append(user_model_score)

                        reuse_pruning = EnsemblePruningReuser(
                            learnware_list=mixture_learnware_list, mode="classification"
                        )
                        reuse_pruning.fit(x_train, y_train)
                        reuse_pruning_predict_y = reuse_pruning.predict(user_data=test_x)
                        reuse_pruning_score = eval_prediction(reuse_pruning_predict_y, test_y)
                        reuse_pruning_score_list.append(reuse_pruning_score)

                    single_score_mat.append([best_acc] * repeated)
                    user_model_score_mat.append(user_model_score_list)
                    pruning_score_mat.append(reuse_pruning_score_list)
                    print(n_label, np.mean(user_model_score_mat[-1]), np.mean(pruning_score_mat[-1]))

                logger.info(f"Saving Curves for User_{i}")
                user_curves_data = (single_score_mat, user_model_score_mat, pruning_score_mat)
                # np.save("./curves/curve" + str(i), user_curves_data)
                with open("./curves/curve" + str(i) + ".pkl", "wb") as f:
                    pickle.dump(user_curves_data, f)
        pruning_curves_data, user_model_curves_data = [], []
        for i in range(n_users):
            with open("./curves/curve" + str(i) + ".pkl", "rb") as f:
                user_curves_data = pickle.load(f)
                (single_score_mat, user_model_score_mat, pruning_score_mat) = user_curves_data
            for i in range(len(single_score_mat)):
                user_model_score_mat[i] = np.mean(user_model_score_mat[i])
                pruning_score_mat[i] = np.mean(pruning_score_mat[i])
            if len(user_model_score_mat) < 6:
                for i in range(6 - len(user_model_score_mat)):
                    user_model_score_mat.append(user_model_score_mat[-1])
                    pruning_score_mat.append(pruning_score_mat[-1])
            user_model_curves_data.append(user_model_score_mat[:6])
            pruning_curves_data.append(pruning_score_mat[:6])
        self._plot_labeled_peformance_curves([user_model_curves_data, pruning_curves_data])

    def _plot_labeled_peformance_curves(self, all_user_curves_data):
        plt.figure(figsize=(10, 6))
        plt.xticks(range(len(n_labeled_list)), n_labeled_list)

        styles = [
            # {"color": "orange", "linestyle": "--", "marker": "s"},
            {"color": "navy", "linestyle": "-", "marker": "o"},
            {"color": "magenta", "linestyle": "-.", "marker": "d"},
        ]

        # labels = ["Single Learnware Reuse", "User Model", "Multiple Learnware Reuse (EnsemblePrune)"]
        labels = ["User Model", "Multiple Learnware Reuse (EnsemblePrune)"]

        user_mat, pruning_mat = all_user_curves_data
        user_mat, pruning_mat = np.array(user_mat), np.array(pruning_mat)
        for mat, style, label in zip([user_mat, pruning_mat], styles, labels):
            mean_curve, std_curve = 1 - np.mean(mat, axis=0), np.std(mat, axis=0)
            plt.plot(mean_curve, **style, label=label)
            plt.fill_between(
                range(len(mean_curve)),
                mean_curve - 0.5 * std_curve,
                mean_curve + 0.5 * std_curve,
                color=style["color"],
                alpha=0.2,
            )

        plt.xlabel("Labeled Data Size")
        plt.ylabel("1 - Accuracy")
        plt.title(f"Text Limited Labeled Data")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("figs", f"text_labeled_curves.png"), bbox_inches="tight", dpi=700)


if __name__ == "__main__":
    fire.Fire(TextDatasetWorkflow)
