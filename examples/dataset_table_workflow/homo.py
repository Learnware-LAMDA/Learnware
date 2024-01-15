import os
import warnings

import numpy as np
from base import TableWorkflow
from config import homo_n_labeled_list, homo_n_repeat_list
from methods import loss_func_rmse
from utils import Recorder, plot_performance_curves

from learnware.logger import get_module_logger
from learnware.market import BaseUserInfo
from learnware.reuse import AveragingReuser, JobSelectorReuser
from learnware.specification import generate_stat_spec

warnings.filterwarnings("ignore")
logger = get_module_logger("homo_table", level="INFO")


class HomogeneousDatasetWorkflow(TableWorkflow):
    def unlabeled_homo_table_example(self):
        logger.info("Total Item: %d" % (len(self.market)))
        learnware_rmse_list = []
        single_score_list = []
        job_selector_score_list = []
        ensemble_score_list = []
        all_learnwares = self.market.get_learnwares()

        user = self.benchmark.name
        for idx in range(self.benchmark.user_num):
            test_x, test_y = self.benchmark.get_test_data(user_ids=idx)
            test_x, test_y = test_x.values, test_y.values
            user_stat_spec = generate_stat_spec(type="table", X=test_x)
            user_info = BaseUserInfo(semantic_spec=self.user_semantic, stat_info={user_stat_spec.type: user_stat_spec})
            logger.info(f"Searching Market for user: {user}_{idx}")

            search_result = self.market.search_learnware(user_info, max_search_num=2)
            single_result = search_result.get_single_results()
            multiple_result = search_result.get_multiple_results()

            logger.info(f"search result of user {user}_{idx}:")
            logger.info(
                f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
            )

            pred_y = single_result[0].learnware.predict(test_x)
            single_score_list.append(loss_func_rmse(pred_y, test_y))

            rmse_list = []
            for learnware in all_learnwares:
                semantic_spec = learnware.specification.get_semantic_spec()
                if semantic_spec["Input"]["Dimension"] == test_x.shape[1]:
                    pred_y = learnware.predict(test_x)
                    rmse_list.append(loss_func_rmse(pred_y, test_y))
            logger.info(
                f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, rmse: {single_score_list[-1]}"
            )

            if len(multiple_result) > 0:
                mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                logger.info(f"mixture_score: {multiple_result[0].score}, mixture_learnware: {mixture_id}")
                mixture_learnware_list = multiple_result[0].learnwares
            else:
                mixture_learnware_list = [single_result[0].learnware]

            # test reuse (job selector)
            reuse_baseline = JobSelectorReuser(learnware_list=mixture_learnware_list, herding_num=100)
            reuse_predict = reuse_baseline.predict(user_data=test_x)
            reuse_score = loss_func_rmse(reuse_predict, test_y)
            job_selector_score_list.append(reuse_score)
            logger.info(f"mixture reuse rmse (job selector): {reuse_score}")

            # test reuse (ensemble)
            reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="mean")
            ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
            ensemble_score = loss_func_rmse(ensemble_predict_y, test_y)
            ensemble_score_list.append(ensemble_score)
            logger.info(f"mixture reuse rmse (ensemble): {ensemble_score}")

            learnware_rmse_list.append(rmse_list)

        single_list = np.array(learnware_rmse_list)
        avg_score_list = [np.mean(lst, axis=0) for lst in single_list]
        oracle_score_list = [np.min(lst, axis=0) for lst in single_list]

        logger.info(
            "RMSE of selected learnware: %.3f +/- %.3f, Average performance: %.3f +/- %.3f, Oracle performace: %.3f +/- %.3f"
            % (
                np.mean(single_score_list),
                np.std(single_score_list),
                np.mean(avg_score_list),
                np.std(avg_score_list),
                np.mean(oracle_score_list),
                np.std(oracle_score_list),
            )
        )
        logger.info(
            "Average Job Selector Reuse Performance: %.3f +/- %.3f"
            % (np.mean(job_selector_score_list), np.std(job_selector_score_list))
        )
        logger.info(
            "Averaging Ensemble Reuse Performance: %.3f +/- %.3f"
            % (np.mean(ensemble_score_list), np.std(ensemble_score_list))
        )

    def labeled_homo_table_example(self, skip_test):
        logger.info("Total Item: %d" % (len(self.market)))
        methods = ["user_model", "homo_single_aug", "homo_ensemble_pruning"]
        recorders = {method: Recorder() for method in methods}
        user = self.benchmark.name

        if not skip_test:
            for idx in range(self.benchmark.user_num):
                test_x, test_y = self.benchmark.get_test_data(user_ids=idx)
                test_x, test_y = test_x.values, test_y.values

                train_x, train_y = self.benchmark.get_train_data(user_ids=idx)
                train_x, train_y = train_x.values, train_y.values
                train_subsets = self.get_train_subsets(homo_n_labeled_list, homo_n_repeat_list, train_x, train_y)

                user_stat_spec = generate_stat_spec(type="table", X=test_x)
                user_info = BaseUserInfo(
                    semantic_spec=self.user_semantic, stat_info={"RKMETableSpecification": user_stat_spec}
                )

                logger.info(f"Searching Market for user: {user}_{idx}")
                search_result = self.market.search_learnware(user_info)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                logger.info(f"search result of user {user}_{idx}:")
                logger.info(
                    f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
                )

                if len(multiple_result) > 0:
                    mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                    logger.info(f"mixture_score: {multiple_result[0].score}, mixture_learnware: {mixture_id}")
                    mixture_learnware_list = multiple_result[0].learnwares
                else:
                    mixture_learnware_list = [single_result[0].learnware]

                test_info = {
                    "user": user,
                    "idx": idx,
                    "train_subsets": train_subsets,
                    "test_x": test_x,
                    "test_y": test_y,
                }
                common_config = {"learnwares": mixture_learnware_list}
                method_configs = {
                    "user_model": {"dataset": self.benchmark.name, "model_type": "lgb"},
                    "homo_single_aug": {"single_learnware": [single_result[0].learnware]},
                    "homo_ensemble_pruning": common_config,
                }

                for method_name in methods:
                    logger.info(f"Testing method {method_name}")
                    test_info["method_name"] = method_name
                    test_info.update(method_configs[method_name])
                    self.test_method(test_info, recorders, loss_func=loss_func_rmse)

            for method, recorder in recorders.items():
                recorder.save(os.path.join(self.curves_result_path, f"{user}/{user}_{method}_performance.json"))

        plot_performance_curves(
            self.curves_result_path, user, recorders, task="Homo", n_labeled_list=homo_n_labeled_list
        )
