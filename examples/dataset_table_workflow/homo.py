import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
from matplotlib import pyplot as plt
from functools import partial
import learnware.specification as specification
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import AveragingReuser, JobSelectorReuser, EnsemblePruningReuser

from benchmarks import DataLoader
from base import TableWorkflow, user_semantic
from methods import *
from config import n_labeled_list, n_repeat_list
from utils import Recorder, plot_performance_curves

logger = get_module_logger("corporacion_test", level="INFO")
learnware_market = ["corporacion_bojan", "corporacion_lee", "corporacion_lingzhi"]
users = ["corporacion_lingzhi"]

class CorporacionDatasetWorkflow(TableWorkflow):
    def __init__(self, reload_market=False, regenerate_flag=False):
        super(CorporacionDatasetWorkflow, self).__init__(learnware_market)
        self.curves_result_path = os.path.join(self.result_path, "curves")
        self.figs_result_path = os.path.join(self.result_path, "figs")

        os.makedirs(self.curves_result_path, exist_ok=True)
        os.makedirs(self.figs_result_path, exist_ok=True)

        if reload_market:
            self.prepare_market(name="easy", market_id="corporacion", regenerate_flag=regenerate_flag)

    def test_homo_unlabeled(self):
        corporacion_market = instantiate_learnware_market(market_id="corporacion")
        logger.info("Total Item: %d" % len(corporacion_market))

        learnware_rmse_list = defaultdict(list)
        job_selector_score_list = defaultdict(list)
        ensemble_score_list = defaultdict(list)
        pruning_score_list = defaultdict(list)

        for user in learnware_market:
            corporacion = DataLoader(user)
            idx_list = corporacion.get_shop_ids()
            for idx in idx_list:
                _, _, test_x, test_y, _ = corporacion.get_raw_data(idx)
                user_stat_spec = specification.RKMETableSpecification()
                user_stat_spec.generate_stat_spec_from_data(X=test_x)
                user_info = BaseUserInfo(
                    semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_stat_spec}
                )
                logger.info(f"Searching Market for user: {user}_{idx}")

                search_result = corporacion_market.search_learnware(user_info, max_search_num=10)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                logger.info(f"search result of user {user}_{idx}:")
                logger.info(
                    f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
                )

                l = len(single_result)
                rmse_list = []
                for idx in range(l):
                    learnware = single_result[idx].learnware
                    pred_y = learnware.predict(test_x)
                    rmse_list.append(loss_func_mse(pred_y, test_y))
                logger.info(
                    f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, rmse: {rmse_list[0]}"
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
                reuse_score = loss_func_mse(reuse_predict, test_y)
                job_selector_score_list[user].append(reuse_score)
                logger.info(f"mixture reuse rmse (job selector): {reuse_score}")

                # test reuse (ensemble)
                reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="mean")
                ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
                ensemble_score = loss_func_mse(ensemble_predict_y, test_y)
                ensemble_score_list[user].append(ensemble_score)
                logger.info(f"mixture reuse rmse (ensemble): {ensemble_score}")

                # test reuse (ensemblePruning)
                reuse_pruning = EnsemblePruningReuser(learnware_list=mixture_learnware_list, mode="regression")
                pruning_predict_y = reuse_pruning.predict(user_data=test_x)
                pruning_score = loss_func_mse(pruning_predict_y, test_y)
                pruning_score_list[user].append(pruning_score)
                logger.info(f"mixture reuse rmse (ensemble Pruning): {pruning_score}\n")

                learnware_rmse_list[user].append(rmse_list)

        for user in learnware_market:
            logger.info(f"User Dataset: {user}")

            single_list = np.array(learnware_rmse_list[user])
            select_score_list = [lst[0] for lst in single_list]
            avg_score_list = [np.mean(lst, axis=0) for lst in single_list]
            oracle_score_list = [np.min(lst, axis=0) for lst in single_list]

            logger.info(
                "RMSE of selected learnware: %.3f +/- %.3f, Average performance: %.3f +/- %.3f, Oracle performace: %.3f +/- %.3f"
                % (
                    np.mean(select_score_list),
                    np.std(select_score_list),
                    np.mean(avg_score_list),
                    np.std(avg_score_list),
                    np.mean(oracle_score_list),
                    np.std(oracle_score_list),
                )
            )
            logger.info(
                "Average Job Selector Reuse Performance: %.3f +/- %.3f"
                % (np.mean(job_selector_score_list[user]), np.std(job_selector_score_list[user]))
            )
            logger.info(
                "Averaging Ensemble Reuse Performance: %.3f +/- %.3f"
                % (np.mean(ensemble_score_list[user]), np.std(ensemble_score_list[user]))
            )
            logger.info(
                "Selective Ensemble Reuse Performance: %.3f +/- %.3f"
                % (np.mean(pruning_score_list[user]), np.std(pruning_score_list[user]))
            )

    def test_homo_labeled(self):
        corporacion_market = instantiate_learnware_market(market_id="corporacion")
        logger.info("Total Item: %d" % len(corporacion_market))

        methods = ["user_model", "homo_single_aug", "homo_multiple_aug", "homo_multiple_avg", "homo_ensemble_pruning"]
        recorders = {method: Recorder() for method in methods}

        methods_to_retest = []

        for user in users:
            data_loader = DataLoader(user)
            idx_list = data_loader.get_shop_ids()
            for idx in idx_list:
                _, _, test_x, test_y, _ = data_loader.get_raw_data(idx)
                train_subsets = data_loader.get_labeled_training_data(
                    idx,
                    size_list=n_labeled_list,
                    n_repeat_list=n_repeat_list
                )

                user_stat_spec = specification.RKMETableSpecification()
                user_stat_spec.generate_stat_spec_from_data(X=test_x)
                user_info = BaseUserInfo(
                    semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_stat_spec}
                )
                logger.info(f"Searching Market for user: {user}_{idx}")

                search_result = corporacion_market.search_learnware(user_info, max_search_num=10)
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

                test_info = {"user": user, "idx": idx, "train_subsets": train_subsets, "test_x": test_x, "test_y": test_y}
                common_config = {"multiple_learnwares": mixture_learnware_list}
                method_configs = {
                    "user_model": {"data_loader": data_loader},
                    "homo_single_aug": {"single_learnware": [single_result[0].learnware]},
                    "homo_multiple_aug": common_config,
                    "homo_multiple_avg": common_config,
                    "homo_ensemble_pruning": common_config
                }

                for method_name in methods:
                    # self.test_method(method_name, HeteroScoringMethods.__dict__[f"{method_name}_score"], recorders, test_info)
                    logger.info(f"Testing method {method_name}")
                    test_info["method_name"] = method_name
                    test_info["force"] = method_name in methods_to_retest
                    test_info.update(method_configs[method_name])
                    self.test_method(test_info, recorders, loss_func=loss_func_mse)
            
            for method, recorder in recorders.items():
                recorder.save(os.path.join(self.curves_result_path, f"{user}_{method}_performance.json"))
            
            methods_to_plot = ["user_model", "homo_single_aug", "homo_ensemble_pruning"]
            plot_performance_curves(user, {method: recorders[method] for method in methods_to_plot}, task="Homo", n_labeled_list=n_labeled_list)