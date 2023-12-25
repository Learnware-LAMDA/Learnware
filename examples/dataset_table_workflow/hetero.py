import os
import warnings
from collections import defaultdict
from functools import partial
warnings.filterwarnings("ignore")

import json
import numpy as np
from matplotlib import pyplot as plt
import learnware.specification as specification
from learnware.logger import get_module_logger
from learnware.market import instantiate_learnware_market, BaseUserInfo
from learnware.reuse import AveragingReuser, FeatureAlignLearnware
from multiprocessing import Pool

from benchmarks import DataLoader
from examples.dataset_table_workflow.methods import *
from base import TableWorkflow, user_semantic
from config import align_model_params
from utils import Recorder, plot_performance_curves, analyze_performance

logger = get_module_logger("hetero_test", level="INFO")
learnware_market = ["pfs_default", "pfs_denis", "corporacion_bojan", "corporacion_lee", "corporacion_lingzhi"]
default_users = ["m5_default"] # "m5_default", "m5_kkiller", "m5_rana"
n_labeled_list = [100, 200, 500, 1000, 2000]
n_repeat_list = [10, 10, 10, 3, 3]

class HeterogeneousWorkflow(TableWorkflow):
    def __init__(self, reload_market=False, regenerate_flag=False):
        super(HeterogeneousWorkflow, self).__init__(learnware_market)
        self.curves_result_path = os.path.join(self.result_path, 'curves')
        self.unlabeled_res_path = os.path.join(self.result_path, 'unlabeled')
        self.figs_result_path = os.path.join(self.result_path, "figs")

        os.makedirs(self.curves_result_path, exist_ok=True)
        os.makedirs(self.unlabeled_res_path, exist_ok=True)
        os.makedirs(self.figs_result_path, exist_ok=True)

        if reload_market:
            self.prepare_market(name="hetero", market_id="heterogeneous", regenerate_flag=regenerate_flag)

    def test_hetero_unlabeled(self):
        hetero_market = instantiate_learnware_market(market_id="heterogeneous", name="hetero")
        logger.info("Total Item: %d" % len(hetero_market))
        
        select_list = defaultdict(list)
        avg_list = defaultdict(list)
        oracle_list = defaultdict(list)
        improve_list = defaultdict(list)
        ensemble_score_list = defaultdict(list)
        user_model_score_lists = defaultdict(list)

        user_model_method = partial(self._limited_data, HeteroMethods.user_model_score)

        for user in default_users:
            user_unlabeld_res_path = os.path.join(self.unlabeled_res_path, f"{user}.json")
            if os.path.exists(user_unlabeld_res_path):
                with open(user_unlabeld_res_path, "rb") as file:
                    unlabeld_res = json.load(file)
                select_list[user] = unlabeld_res["select_list"]
                avg_list[user] = unlabeld_res["avg_list"]
                oracle_list[user] = unlabeld_res["oracle_list"]
                ensemble_score_list[user] = unlabeld_res["ensemble_score_list"]
                user_model_score_lists[user] = unlabeld_res["user_model_score_lists"]
            else:
                data_loader = DataLoader(user)
                idx_list = data_loader.get_shop_ids()
                for idx in idx_list:
                    _, _, test_x, test_y, feature_descriptions = data_loader.get_raw_data(idx)
                    train_x_list, _ = data_loader.get_labeled_training_data(idx, size_list=n_labeled_list, n_repeat_list=n_repeat_list)
                    user_stat_spec = specification.RKMETableSpecification()
                    user_stat_spec.generate_stat_spec_from_data(X=test_x)

                    feature_dim = len(feature_descriptions)
                    feature_descriptions_dict = {str(i): feature_descriptions[i] for i in range(feature_dim)}
                    input_description = {
                        "Dimension": feature_dim,
                        "Description": feature_descriptions_dict
                    }
                    user_semantic["Input"] = input_description
                    user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_stat_spec})
                    logger.info(f"Searching Market for user: {user}_{idx}")
                    
                    search_result = hetero_market.search_learnware(user_info)
                    single_result = search_result.get_single_results()
                    multiple_result = search_result.get_multiple_results()
                    
                    logger.info(f"hetero search result of user {user}_{idx}:")
                    logger.info(
                        f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
                    )

                    l = len(single_result)
                    rmse_list = []
                    for idx in range(l):
                        hetero_learnware = FeatureAlignLearnware(single_result[idx].learnware, **align_model_params)
                        hetero_learnware.align(user_rkme=user_stat_spec)
                        pred_y = hetero_learnware.predict(test_x)
                        rmse_list.append(loss_func_rmse(pred_y, test_y))
                    
                    logger.info(
                        f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, rmse: {rmse_list[0]}"
                    )
                    
                    if len(multiple_result) > 0:
                        mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                        logger.info(f"mixture_score: {multiple_result[0].score}, mixture_learnware: {mixture_id}")
                        mixture_learnware_list = []
                        for learnware in multiple_result[0].learnwares:
                            hetero_learnware = FeatureAlignLearnware(learnware, **align_model_params)
                            hetero_learnware.align(user_rkme=user_stat_spec)
                            mixture_learnware_list.append(hetero_learnware)
                    else:
                        hetero_learnware = FeatureAlignLearnware(single_result[0].learnware, **align_model_params)
                        hetero_learnware.align(user_rkme=user_stat_spec)
                        mixture_learnware_list = [hetero_learnware]
                    
                    # test user model
                    user_model_score_list = user_model_method(train_x_list, test_x, test_y, data_loader)

                    # test reuse (ensemble)
                    reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="mean")
                    ensemble_predict_y = reuse_ensemble.predict(user_data=test_x)
                    ensemble_score = loss_func_rmse(ensemble_predict_y, test_y)
                    ensemble_score_list[user].append(ensemble_score)
                    logger.info(f"mixture reuse rmse (ensemble): {ensemble_score}")

                    select_list[user].append(rmse_list[0])
                    avg_list[user].append(np.mean(rmse_list))
                    oracle_list[user].append(np.min(rmse_list))
                    improve_list[user].append((np.mean(rmse_list) - rmse_list[0]) / np.mean(rmse_list))
                    user_model_score_lists[user].append(user_model_score_list)
                
                logger.info(f"Saving unlabeled results for User: {user}")
                res = {
                    "select_list": select_list[user],
                    "avg_list": avg_list[user],
                    "oracle_list": oracle_list[user],
                    "ensemble_score_list": ensemble_score_list[user],
                    "user_model_score_lists": user_model_score_lists[user]
                }
                with open(user_unlabeld_res_path, "w") as file:
                    json.dump(res, file, indent=4)

        for user in default_users:
            logger.info(f"User Dataset: {user}")
            logger.info(
                "RMSE of selected learnware: %.3f +/- %.3f, Average performance: %.3f +/- %.3f, Oracle performace: %.3f +/- %.3f"
                % (np.mean(select_list[user]), np.std(select_list[user]), np.mean(avg_list[user]), np.std(avg_list[user]), np.mean(oracle_list[user]), np.std(oracle_list[user]))
            )
            logger.info(
                "Averaging Ensemble Reuse Performance: %.3f +/- %.3f"
                % (np.mean(ensemble_score_list[user]), np.std(ensemble_score_list[user]))
            )
            for idx, n_labeled in enumerate(n_labeled_list):
                n_labeled_score_list = [lst[idx] for lst in user_model_score_lists[user]]
                logger.info(
                    "User Model with %d data: %.3f +/ %.3f"
                    % (n_labeled, np.mean(n_labeled_score_list), np.std(n_labeled_score_list))
                )

    def test_hetero_labeled(self):
        hetero_market = instantiate_learnware_market(market_id="heterogeneous", name="hetero")
        logger.info("Total Items: %d" % len(hetero_market))

        methods = ["user_model", "hetero_single_aug", "hetero_multiple_aug", "hetero_multiple_avg", "hetero_ensemble_pruning"]
        methods_to_test = ["hetero_multiple_avg", "hetero_ensemble_pruning"]
        recorders = {method: Recorder() for method in methods + ["select_score", "oracle_score", "mean_score"]}

        for user in default_users:
            data_loader = DataLoader(user)
            idx_list = data_loader.get_shop_ids()
            for idx in idx_list:
                _, _, test_x, test_y, feature_descriptions = data_loader.get_raw_data(idx)
                train_subsets = data_loader.get_labeled_training_data(
                    idx,
                    size_list=n_labeled_list,
                    n_repeat_list=n_repeat_list
                )
                user_stat_spec = specification.RKMETableSpecification()
                user_stat_spec.generate_stat_spec(X=test_x)

                input_description = {
                    "Dimension": len(feature_descriptions),
                    "Description": {str(i): feature_descriptions[i] for i in range(len(feature_descriptions))}
                }
                user_semantic["Input"] = input_description
                user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMETableSpecification": user_stat_spec})
                logger.info(f"Searching Market for user: {user}_{idx}")

                search_result = hetero_market.search_learnware(user_info, max_search_num=10)
                single_result = search_result.get_single_results()
                multiple_result = search_result.get_multiple_results()

                if len(multiple_result) > 0:
                    mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                    logger.info(f"Mixture score: {multiple_result[0].score}, Mixture learnware: {mixture_id}")
                    mixture_learnware_list = multiple_result[0].learnwares
                else:
                    mixture_learnware_list = [single_result[0].learnware]
                
                logger.info(f"Hetero search result of user {user}_{idx}: mixture learnware num: {len(mixture_learnware_list)}")

                test_info = {"user": user, "idx": idx, "train_subsets": train_subsets, "test_x": test_x, "test_y": test_y}
                common_config = {"user_rkme": user_stat_spec, "multiple_learnwares": mixture_learnware_list}
                method_configs = {
                    "user_model": {"data_loader": data_loader},
                    "hetero_single_aug": {"user_rkme": user_stat_spec, "learnwares": [single_result[0].learnware]},
                    "hetero_multiple_aug": common_config,
                    "hetero_multiple_avg": common_config,
                    "hetero_ensemble_pruning": common_config
                }

                for method_name in methods:
                    test_info["method_name"] = method_name
                    test_info["force"] = method_name in methods_to_test
                    test_info.update(method_configs[method_name])
                    self.test_method(test_info, recorders, loss_func=loss_func_rmse)
            
            for method, recorder in recorders.items():
                recorder.save(os.path.join(self.curves_result_path, f"{user}/{user}_{method}_performance.json"))
                
            methods_to_plot = ["user_model", "select_score", "hetero_multiple_avg", "hetero_ensemble_pruning"]
            plot_performance_curves(user, {method: recorders[method] for method in methods_to_plot}, n_labeled_list=n_labeled_list)
            