import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from learnware.logger import get_module_logger
from learnware.specification import generate_stat_spec
from learnware.market import BaseUserInfo
from learnware.reuse import AveragingReuser, FeatureAlignLearnware

from methods import *
from base import TableWorkflow
from config import align_model_params, user_semantic, hetero_n_labeled_list, hetero_n_repeat_list
from utils import Recorder, plot_performance_curves

logger = get_module_logger("hetero_test", level="INFO")
n_labeled_list = [10, 30, 50, 75, 100, 200]
n_repeat_list = [10, 10, 10, 10, 10, 10]

class HeterogeneousDatasetWorkflow(TableWorkflow):
    def unlabeled_hetero_table_example(self):
        logger.info("Total Item: %d" % len(self.market))
        learnware_rmse_list = []
        single_score_list = []
        ensemble_score_list = []
        all_learnwares = self.market.get_learnwares()
        
        user = self.benchmark.name
        for idx in range(self.benchmark.user_num):
            test_x, test_y = self.benchmark.get_test_data(user_ids=idx)
            test_x, test_y, feature_descriptions = test_x.values, test_y.values, test_x.columns
            user_stat_spec = generate_stat_spec(type="table", X=test_x)
            input_description = {
                "Dimension": len(feature_descriptions),
                "Description": {str(i): feature_descriptions[i] for i in range(len(feature_descriptions))}
            }
            user_semantic["Input"] = input_description
            user_info = BaseUserInfo(
                semantic_spec=user_semantic, stat_info={user_stat_spec.type: user_stat_spec}
            )
            logger.info(f"Searching Market for user: {user}_{idx}")
            
            search_result = self.market.search_learnware(user_info)
            single_result = search_result.get_single_results()
            multiple_result = search_result.get_multiple_results()
            
            logger.info(f"hetero search result of user {user}_{idx}:")
            logger.info(
                f"single model num: {len(single_result)}, max_score: {single_result[0].score}, min_score: {single_result[-1].score}"
            )
            
            single_hetero_learnware = FeatureAlignLearnware(single_result[0].learnware, **align_model_params)
            single_hetero_learnware.align(user_rkme=user_stat_spec)
            pred_y = single_hetero_learnware.predict(test_x)
            single_score_list.append(loss_func_rmse(pred_y, test_y))

            rmse_list = []
            for learnware in all_learnwares:
                hetero_learnware = FeatureAlignLearnware(learnware, **align_model_params)
                hetero_learnware.align(user_rkme=user_stat_spec)
                pred_y = hetero_learnware.predict(test_x)
                rmse_list.append(loss_func_rmse(pred_y, test_y)) 
            logger.info(
                f"Top1-score: {single_result[0].score}, learnware_id: {single_result[0].learnware.id}, rmse: {single_score_list[0]}"
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
            "Averaging Ensemble Reuse Performance: %.3f +/- %.3f"
            % (np.mean(ensemble_score_list), np.std(ensemble_score_list))
        )

    def labeled_hetero_table_example(self):
        logger.info("Total Items: %d" % len(self.market))
        methods = ["user_model", "hetero_single_aug", "hetero_multiple_avg", "hetero_ensemble_pruning"]
        recorders = {method: Recorder() for method in methods + ["select_score", "oracle_score", "mean_score"]}

        user = self.benchmark.name
        for idx in range(self.benchmark.user_num):
            test_x, test_y = self.benchmark.get_test_data(user_ids=idx)
            test_x, test_y = test_x.values, test_y.values
            
            train_x, train_y = self.benchmark.get_train_data(user_ids=idx)
            train_x, train_y, feature_descriptions = train_x.values, train_y.values, train_x.columns
            train_subsets = self.get_train_subsets(hetero_n_labeled_list, hetero_n_repeat_list, idx, train_x, train_y)
            
            user_stat_spec = generate_stat_spec(type="table", X=test_x)
            input_description = {
                "Dimension": len(feature_descriptions),
                "Description": {str(i): feature_descriptions[i] for i in range(len(feature_descriptions))}
            }
            user_semantic["Input"] = input_description
            user_info = BaseUserInfo(
                semantic_spec=user_semantic, stat_info={user_stat_spec.type: user_stat_spec}
            )
            logger.info(f"Searching Market for user: {user}_{idx}")

            search_result = self.market.search_learnware(user_info)
            single_result = search_result.get_single_results()
            multiple_result = search_result.get_multiple_results()
            
            rank_map = {item.learnware.id: index for index, item in enumerate(single_result)}
            all_learnwares = self.market.get_learnwares()
            all_learnwares.sort(key=lambda learnware: rank_map.get(learnware.id, float('inf')))

            if len(multiple_result) > 0:
                mixture_id = " ".join([learnware.id for learnware in multiple_result[0].learnwares])
                logger.info(f"Mixture score: {multiple_result[0].score}, Mixture learnware: {mixture_id}")
                mixture_learnware_list = multiple_result[0].learnwares
            else:
                mixture_learnware_list = [single_result[0].learnware]
            
            logger.info(f"Hetero search result of user {user}_{idx}: mixture learnware num: {len(mixture_learnware_list)}")

            test_info = {"user": user, "idx": idx, "train_subsets": train_subsets, "test_x": test_x, "test_y": test_y, "n_labeled_list": hetero_n_labeled_list}
            common_config = {"user_rkme": user_stat_spec, "learnwares": mixture_learnware_list}
            method_configs = {
                "user_model": {"dataset": self.benchmark.name, "model_type": "lgb"},
                "hetero_single_aug": {"user_rkme": user_stat_spec, "learnwares": all_learnwares},
                "hetero_multiple_aug": common_config,
                "hetero_multiple_avg": common_config,
                "hetero_ensemble_pruning": common_config
            }

            for method_name in methods:
                logger.info(f"Testing method {method_name}")
                test_info["method_name"] = method_name
                test_info.update(method_configs[method_name])
                self.test_method(test_info, recorders, loss_func=loss_func_rmse)
        
        for method, recorder in recorders.items():
            recorder.save(os.path.join(self.curves_result_path, f"{user}/{user}_{method}_performance.json"))
                
        methods_to_plot = ["user_model", "select_score", "hetero_multiple_avg", "hetero_ensemble_pruning"]
        plot_performance_curves(self.curves_result_path, user, {method: recorders[method] for method in methods_to_plot}, task="Hetero", n_labeled_list=n_labeled_list)