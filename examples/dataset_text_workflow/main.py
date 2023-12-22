import os
import fire
import time
import random
import pickle
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from learnware.client import LearnwareClient
from learnware.logger import get_module_logger
from learnware.specification import RKMETextSpecification
from learnware.tests.benchmarks import LearnwareBenchmark
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import JobSelectorReuser, AveragingReuser, EnsemblePruningReuser
from config import text_benchmark_config

logger = get_module_logger("text_workflow", level="INFO")


def train(X, y):
    # Train Uploaders' models
    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_tfidf, y)

    return vectorizer, clf


def eval_prediction(pred_y, target_y):
    if not isinstance(pred_y, np.ndarray):
        pred_y = pred_y.detach().cpu().numpy()
    if len(pred_y.shape) == 1:
        predicted = np.array(pred_y)
    else:
        predicted = np.argmax(pred_y, 1)
    annos = np.array(target_y)

    total = predicted.shape[0]
    correct = (predicted == annos).sum().item()

    return correct / total


class TextDatasetWorkflow:
    def prepare_market(self, rebuild=False):
        client = LearnwareClient()
        self.text_benchmark = LearnwareBenchmark().get_benchmark(text_benchmark_config)
        self.text_market = instantiate_learnware_market(market_id=self.text_benchmark.name, rebuild=rebuild)
        self.user_semantic = client.get_semantic_specification(self.text_benchmark.learnware_ids[0])

        if len(self.text_market) == 0 or rebuild == True:
            for learnware_id in self.text_benchmark.learnware_ids:
                with tempfile.TemporaryDirectory(prefix="text_benchmark_") as tempdir:
                    zip_path = os.path.join(tempdir, f"{learnware_id}.zip")
                    for i in range(20):
                        try:
                            semantic_spec = client.get_semantic_specification(learnware_id)
                            client.download_learnware(learnware_id, zip_path)
                            break
                        except:
                            time.sleep(1)
                            continue
                    self.text_market.add_learnware(zip_path, semantic_spec)

        logger.info("Total Item: %d" % (len(self.text_market)))

    def test_unlabeled(self, rebuild=False):
        self.prepare_market(rebuild)

        select_list = []
        avg_list = []
        best_list = []
        improve_list = []
        job_selector_score_list = []
        ensemble_score_list = []
        all_learnwares = self.text_market.get_learnwares()

        for i in range(self.text_benchmark.user_num):
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

    def test_labeled(self, rebuild=False, train_flag=True):
        self.n_labeled_list = [100, 200, 500, 1000, 2000, 4000]
        self.repeated_list = [10, 10, 10, 3, 3, 3]
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.fig_path = os.path.join(self.root_path, "figs")
        self.curve_path = os.path.join(self.root_path, "curves")

        if train_flag:
            self.prepare_market(rebuild)
            os.makedirs(self.fig_path, exist_ok=True)
            os.makedirs(self.curve_path, exist_ok=True)

            for i in range(self.text_benchmark.user_num):
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

                for n_label, repeated in zip(self.n_labeled_list, self.repeated_list):
                    user_model_score_list, reuse_pruning_score_list = [], []
                    if n_label > len(train_x):
                        n_label = len(train_x)
                    for _ in range(repeated):
                        # x_train, y_train = train_x[:n_label], train_y[:n_label]
                        x_train, y_train = zip(*random.sample(list(zip(train_x, train_y)), k=n_label))
                        x_train = list(x_train)
                        y_train = np.array(list(y_train))

                        modelv, modell = train(x_train, y_train)
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
                with open(os.path.join(self.curve_path, f"curve{str(i)}.pkl"), "wb") as f:
                    pickle.dump(user_curves_data, f)

        pruning_curves_data, user_model_curves_data = [], []
        for i in range(self.text_benchmark.user_num):
            with open(os.path.join(self.curve_path, f"curve{str(i)}.pkl"), "rb") as f:
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
        plt.xticks(range(len(self.n_labeled_list)), self.n_labeled_list)

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
        plt.savefig(os.path.join(self.fig_path, "text_labeled_curves.png"), bbox_inches="tight", dpi=700)


if __name__ == "__main__":
    fire.Fire(TextDatasetWorkflow)
