import os
import pickle
import random
import tempfile
import time

import fire
import matplotlib.pyplot as plt
import numpy as np
from config import text_benchmark_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.market import BaseUserInfo, instantiate_learnware_market
from learnware.reuse import AveragingReuser, EnsemblePruningReuser, JobSelectorReuser
from learnware.specification import RKMETextSpecification
from learnware.tests.benchmarks import LearnwareBenchmark

logger = get_module_logger("text_workflow", level="INFO")


class TextDatasetWorkflow:
    @staticmethod
    def _train_model(X, y):
        vectorizer = TfidfVectorizer(stop_words="english")
        X_tfidf = vectorizer.fit_transform(X)
        clf = MultinomialNB(alpha=0.1)
        clf.fit(X_tfidf, y)
        return vectorizer, clf

    @staticmethod
    def _eval_prediction(pred_y, target_y):
        if not isinstance(pred_y, np.ndarray):
            pred_y = pred_y.detach().cpu().numpy()

        pred_y = np.array(pred_y) if len(pred_y.shape) == 1 else np.argmax(pred_y, 1)
        target_y = np.array(target_y)
        return accuracy_score(target_y, pred_y)

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
        plt.title("Results on Text Experimental Scenario", fontsize=16)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, "text_labeled_curves.svg"), bbox_inches="tight", dpi=700)

    def _prepare_market(self, rebuild=False):
        client = LearnwareClient()
        self.text_benchmark = LearnwareBenchmark().get_benchmark(text_benchmark_config)
        self.text_market = instantiate_learnware_market(market_id=self.text_benchmark.name, rebuild=rebuild)
        self.user_semantic = client.get_semantic_specification(self.text_benchmark.learnware_ids[0])
        self.user_semantic["Name"]["Values"] = ""

        if len(self.text_market) == 0 or rebuild is True:
            for learnware_id in self.text_benchmark.learnware_ids:
                with tempfile.TemporaryDirectory(prefix="text_benchmark_") as tempdir:
                    zip_path = os.path.join(tempdir, f"{learnware_id}.zip")
                    for i in range(20):
                        try:
                            semantic_spec = client.get_semantic_specification(learnware_id)
                            client.download_learnware(learnware_id, zip_path)
                            self.text_market.add_learnware(zip_path, semantic_spec)
                            break
                        except Exception:
                            time.sleep(1)
                            continue

        logger.info("Total Item: %d" % (len(self.text_market)))

    def unlabeled_text_example(self, rebuild=False):
        self._prepare_market(rebuild)

        select_list = []
        avg_list = []
        best_list = []
        improve_list = []
        job_selector_score_list = []
        ensemble_score_list = []
        all_learnwares = self.text_market.get_learnwares()

        for i in range(text_benchmark_config.user_num):
            user_data, user_label = self.text_benchmark.get_test_data(user_ids=i)

            user_stat_spec = RKMETextSpecification()
            user_stat_spec.generate_stat_spec_from_data(X=user_data)
            user_info = BaseUserInfo(
                semantic_spec=self.user_semantic, stat_info={"RKMETextSpecification": user_stat_spec}
            )
            logger.info("Searching Market for user: %d" % (i))

            search_result = self.text_market.search_learnware(user_info)
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
                acc = self._eval_prediction(pred_y, user_label)
                acc_list.append(acc)

            learnware = single_result[0].learnware
            pred_y = learnware.predict(user_data)
            best_acc = self._eval_prediction(pred_y, user_label)
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
            reuse_score = self._eval_prediction(reuse_predict, user_label)
            job_selector_score_list.append(reuse_score)
            print(f"mixture reuse accuracy (job selector): {reuse_score}")

            # test reuse (ensemble)
            reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote_by_label")
            ensemble_predict_y = reuse_ensemble.predict(user_data=user_data)
            ensemble_score = self._eval_prediction(ensemble_predict_y, user_label)
            ensemble_score_list.append(ensemble_score)
            print(f"mixture reuse accuracy (ensemble): {ensemble_score}\n")

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

    def labeled_text_example(self, rebuild=False, skip_test=False):
        self.n_labeled_list = [100, 200, 500, 1000, 2000, 4000]
        self.repeated_list = [10, 10, 10, 3, 3, 3]
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.fig_path = os.path.join(self.root_path, "figs")
        self.curve_path = os.path.join(self.root_path, "curves")

        if not skip_test:
            self._prepare_market(rebuild)
            os.makedirs(self.fig_path, exist_ok=True)
            os.makedirs(self.curve_path, exist_ok=True)

            for i in range(text_benchmark_config.user_num):
                user_model_score_mat = []
                pruning_score_mat = []
                single_score_mat = []
                test_x, test_y = self.text_benchmark.get_test_data(user_ids=i)
                test_y = np.array(test_y)

                train_x, train_y = self.text_benchmark.get_train_data(user_ids=i)
                train_y = np.array(train_y)

                user_stat_spec = RKMETextSpecification()
                user_stat_spec.generate_stat_spec_from_data(X=test_x)
                user_info = BaseUserInfo(
                    semantic_spec=self.user_semantic, stat_info={"RKMETextSpecification": user_stat_spec}
                )
                logger.info(f"Searching Market for user_{i}")

                search_result = self.text_market.search_learnware(user_info)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                learnware = single_result[0].learnware
                pred_y = learnware.predict(test_x)
                best_acc = self._eval_prediction(pred_y, test_y)
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

                for n_label, repeated in zip(self.n_labeled_list, self.repeated_list):
                    user_model_score_list, reuse_pruning_score_list = [], []
                    if n_label > len(train_x):
                        n_label = len(train_x)
                    for _ in range(repeated):
                        x_train, y_train = zip(*random.sample(list(zip(train_x, train_y)), k=n_label))
                        x_train = list(x_train)
                        y_train = np.array(list(y_train))

                        modelv, modell = self._train_model(x_train, y_train)
                        user_model_predict_y = modell.predict(modelv.transform(test_x))
                        user_model_score = self._eval_prediction(user_model_predict_y, test_y)
                        user_model_score_list.append(user_model_score)

                        reuse_pruning = EnsemblePruningReuser(
                            learnware_list=mixture_learnware_list, mode="classification"
                        )
                        reuse_pruning.fit(x_train, y_train)
                        reuse_pruning_predict_y = reuse_pruning.predict(user_data=test_x)
                        reuse_pruning_score = self._eval_prediction(reuse_pruning_predict_y, test_y)
                        reuse_pruning_score_list.append(reuse_pruning_score)

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

        pruning_curves_data, user_model_curves_data = [], []
        total_user_model_score_mat = [np.zeros(self.repeated_list[i]) for i in range(len(self.n_labeled_list))]
        total_pruning_score_mat = [np.zeros(self.repeated_list[i]) for i in range(len(self.n_labeled_list))]
        for user_idx in range(text_benchmark_config.user_num):
            with open(os.path.join(self.curve_path, f"curve{str(user_idx)}.pkl"), "rb") as f:
                user_curves_data = pickle.load(f)
                (single_score_mat, user_model_score_mat, pruning_score_mat) = user_curves_data

                for i in range(len(self.n_labeled_list)):
                    total_user_model_score_mat[i] += 1 - np.array(user_model_score_mat[i])
                    total_pruning_score_mat[i] += 1 - np.array(pruning_score_mat[i])

        for i in range(len(self.n_labeled_list)):
            total_user_model_score_mat[i] /= text_benchmark_config.user_num
            total_pruning_score_mat[i] /= text_benchmark_config.user_num
            user_model_curves_data.append(
                (np.mean(total_user_model_score_mat[i]), np.std(total_user_model_score_mat[i]))
            )
            pruning_curves_data.append((np.mean(total_pruning_score_mat[i]), np.std(total_pruning_score_mat[i])))

        self._plot_labeled_peformance_curves([user_model_curves_data, pruning_curves_data])


if __name__ == "__main__":
    fire.Fire(TextDatasetWorkflow)
