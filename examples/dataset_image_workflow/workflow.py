import os
import fire
import time
import torch
import pickle
import random
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from learnware.utils import choose_device
from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.specification import generate_stat_spec
from learnware.tests.benchmarks import LearnwareBenchmark
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser
from model import ConvModel
from utils import train_model, evaluate
from config import image_benchmark_config

logger = get_module_logger("image_workflow", level="INFO")


class ImageDatasetWorkflow:
    def _plot_labeled_peformance_curves(self, all_user_curves_data):
        plt.figure(figsize=(10, 6))
        plt.xticks(range(len(self.n_labeled_list)), self.n_labeled_list)

        styles = [
            {"color": "navy", "linestyle": "-", "marker": "o"},
            {"color": "magenta", "linestyle": "-.", "marker": "d"},
        ]
        labels = ["User Model", "Multiple Learnware Reuse (EnsemblePrune)"]

        user_array, pruning_array = all_user_curves_data
        for array, style, label in zip([user_array, pruning_array], styles, labels):
            mean_curve = np.array([item[0] for item in array])
            std_curve = np.array([item[1] for item in array])
            plt.plot(mean_curve, **style, label=label)
            plt.fill_between(
                range(len(mean_curve)),
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=style["color"],
                alpha=0.2,
            )

        plt.xlabel("Amout of Labeled User Data", fontsize=14)
        plt.ylabel("1 - Accuracy", fontsize=14)
        plt.title(f"Results on Image Experimental Scenario", fontsize=16)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, "image_labeled_curves.svg"), bbox_inches="tight", dpi=700)

    def _prepare_market(self, rebuild=False):
        client = LearnwareClient()
        self.image_benchmark = LearnwareBenchmark().get_benchmark(image_benchmark_config)
        self.image_market = instantiate_learnware_market(market_id=self.image_benchmark.name, rebuild=rebuild)
        self.user_semantic = client.get_semantic_specification(self.image_benchmark.learnware_ids[0])
        self.user_semantic["Name"]["Values"] = ""

        if len(self.image_market) == 0 or rebuild == True:
            for learnware_id in self.image_benchmark.learnware_ids:
                with tempfile.TemporaryDirectory(prefix="image_benchmark_") as tempdir:
                    zip_path = os.path.join(tempdir, f"{learnware_id}.zip")
                    for i in range(20):
                        try:
                            semantic_spec = client.get_semantic_specification(learnware_id)
                            client.download_learnware(learnware_id, zip_path)
                            self.image_market.add_learnware(zip_path, semantic_spec)
                            break
                        except:
                            time.sleep(1)
                            continue

        logger.info("Total Item: %d" % (len(self.image_market)))

    def image_example(self, rebuild=False, skip_test=True):
        np.random.seed(1)
        random.seed(1)
        self.n_labeled_list = [100, 200, 500, 1000, 2000, 4000]
        self.repeated_list = [10, 10, 10, 3, 3, 3]
        device = choose_device(0)

        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.fig_path = os.path.join(self.root_path, "figs")
        self.curve_path = os.path.join(self.root_path, "curves")
        self.model_path = os.path.join(self.root_path, "models")
        os.makedirs(self.fig_path, exist_ok=True)
        os.makedirs(self.curve_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        select_list = []
        avg_list = []
        best_list = []
        improve_list = []
        job_selector_score_list = []
        ensemble_score_list = []

        if not skip_test:
            self._prepare_market(rebuild)
            all_learnwares = self.image_market.get_learnwares()

            for i in range(image_benchmark_config.user_num):
                test_x, test_y = self.image_benchmark.get_test_data(user_ids=i)
                train_x, train_y = self.image_benchmark.get_train_data(user_ids=i)

                test_x = torch.from_numpy(test_x)
                test_y = torch.from_numpy(test_y)
                test_dataset = TensorDataset(test_x, test_y)

                user_stat_spec = generate_stat_spec(type="image", X=test_x, whitening=False)
                user_info = BaseUserInfo(semantic_spec=self.user_semantic, stat_info={user_stat_spec.type: user_stat_spec})
                logger.info("Searching Market for user: %d" % (i))

                search_result = self.image_market.search_learnware(user_info)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                print(f"search result of user{i}:")
                print(
                    f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
                )

                acc_list = []
                for idx in range(len(all_learnwares)):
                    learnware = all_learnwares[idx]
                    loss, acc = evaluate(learnware, test_dataset)
                    acc_list.append(acc)

                learnware = single_result[0].learnware
                best_loss, best_acc = evaluate(learnware, test_dataset)
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
                reuse_job_selector = JobSelectorReuser(learnware_list=mixture_learnware_list, use_herding=False)
                job_loss, job_acc = evaluate(reuse_job_selector, test_dataset)
                job_selector_score_list.append(job_acc)
                print(f"mixture reuse accuracy (job selector): {job_acc}")

                # test reuse (ensemble)
                reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote_by_prob")
                ensemble_loss, ensemble_acc = evaluate(reuse_ensemble, test_dataset)
                ensemble_score_list.append(ensemble_acc)
                print(f"mixture reuse accuracy (ensemble): {ensemble_acc}\n")

                user_model_score_mat = []
                pruning_score_mat = []
                single_score_mat = []

                for n_label, repeated in zip(self.n_labeled_list, self.repeated_list):
                    user_model_score_list, reuse_pruning_score_list = [], []
                    if n_label > len(train_x):
                        n_label = len(train_x)
                    for _ in range(repeated):
                        x_train, y_train = zip(*random.sample(list(zip(train_x, train_y)), k=n_label))
                        x_train = np.array(list(x_train))
                        y_train = np.array(list(y_train))

                        x_train = torch.from_numpy(x_train)
                        y_train = torch.from_numpy(y_train)
                        sampled_dataset = TensorDataset(x_train, y_train)

                        mode_save_path = os.path.abspath(os.path.join(self.model_path, "model.pth"))
                        model = ConvModel(
                            channel=x_train.shape[1], im_size=(x_train.shape[2], x_train.shape[3]), n_random_features=10
                        ).to(device)
                        train_model(
                            model,
                            sampled_dataset,
                            sampled_dataset,
                            mode_save_path,
                            epochs=35,
                            batch_size=128,
                            device=device,
                            verbose=False,
                        )
                        model.load_state_dict(torch.load(mode_save_path))
                        _, user_model_acc = evaluate(model, test_dataset, distribution=True)
                        user_model_score_list.append(user_model_acc)

                        reuse_pruning = EnsemblePruningReuser(learnware_list=mixture_learnware_list, mode="classification")
                        reuse_pruning.fit(x_train, y_train)
                        _, pruning_acc = evaluate(reuse_pruning, test_dataset, distribution=False)
                        reuse_pruning_score_list.append(pruning_acc)

                    single_score_mat.append([best_acc] * repeated)
                    user_model_score_mat.append(user_model_score_list)
                    pruning_score_mat.append(reuse_pruning_score_list)
                    print(
                        f"user_label_num: {n_label}, user_acc: {np.mean(user_model_score_mat[-1])}, pruning_acc: {np.mean(pruning_score_mat[-1])}"
                    )

                logger.info(f"Saving Curves for User_{i}")
                user_curves_data = (single_score_mat, user_model_score_mat, pruning_score_mat)
                with open(os.path.join(self.curve_path, f"curve{str(i)}.pkl"), "wb") as f:
                    pickle.dump(user_curves_data, f)

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

        pruning_curves_data, user_model_curves_data = [], []
        total_user_model_score_mat = [np.zeros(self.repeated_list[i]) for i in range(len(self.n_labeled_list))]
        total_pruning_score_mat = [np.zeros(self.repeated_list[i]) for i in range(len(self.n_labeled_list))]
        for user_idx in range(image_benchmark_config.user_num):
            with open(os.path.join(self.curve_path, f"curve{str(user_idx)}.pkl"), "rb") as f:
                user_curves_data = pickle.load(f)
                (single_score_mat, user_model_score_mat, pruning_score_mat) = user_curves_data

                for i in range(len(self.n_labeled_list)):
                    total_user_model_score_mat[i] += 1 - np.array(user_model_score_mat[i]) / 100
                    total_pruning_score_mat[i] += 1 - np.array(pruning_score_mat[i]) / 100

        for i in range(len(self.n_labeled_list)):
            total_user_model_score_mat[i] /= image_benchmark_config.user_num
            total_pruning_score_mat[i] /= image_benchmark_config.user_num
            user_model_curves_data.append(
                (np.mean(total_user_model_score_mat[i]), np.std(total_user_model_score_mat[i]))
            )
            pruning_curves_data.append((np.mean(total_pruning_score_mat[i]), np.std(total_pruning_score_mat[i])))

        self._plot_labeled_peformance_curves([user_model_curves_data, pruning_curves_data])


if __name__ == "__main__":
    fire.Fire(ImageDatasetWorkflow)
