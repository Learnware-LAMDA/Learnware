import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from learnware.reuse import AveragingReuser, EnsemblePruningReuser, FeatureAugmentReuser, HeteroMapAlignLearnware
from config import align_model_params
from train import train_model


def loss_func_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def user_model_score(x_train, y_train, test_info):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    user_model = train_model(x_train, y_train, x_val, y_val, test_info)
    return user_model


class HomoScoringMethods:
    @staticmethod
    def single_aug_score(x_train, y_train, test_info):
        single_learnware = test_info["single_learnware"]
        reuse_single_augment = FeatureAugmentReuser(single_learnware, mode="regression")
        reuse_single_augment.fit(x_train=x_train, y_train=y_train)
        return reuse_single_augment

    @staticmethod
    def multiple_aug_score(x_train, y_train, test_info):
        multiple_learnwares = test_info["learnwares"]
        reuse_multiple_augment = FeatureAugmentReuser(multiple_learnwares, mode="regression")
        reuse_multiple_augment.fit(x_train=x_train, y_train=y_train)
        return reuse_multiple_augment
    
    @staticmethod
    def multiple_avg_score(x_train, y_train, test_info):
        multiple_learnwares = test_info["learnwares"]
        reuse_multiple_avg = AveragingReuser(multiple_learnwares, mode="mean")
        return reuse_multiple_avg

    @staticmethod
    def multiple_ensemble_pruning_score(x_train, y_train, test_info):
        multiple_learnwares = test_info["learnwares"]
        if len(multiple_learnwares) == 1:
            return multiple_learnwares[0]
        reuse_pruning = EnsemblePruningReuser(multiple_learnwares, mode="regression")
        reuse_pruning.fit(val_X=x_train, val_y=y_train)
        return reuse_pruning


class HeteroMethods:
    @staticmethod
    def create_hetero_learnware_list(learnware_list, user_rkme, x_train, y_train):
        hetero_learnware_list = []
        for learnware in learnware_list:
            hetero_learnware = HeteroMapAlignLearnware(learnware, mode="regression", **align_model_params)
            hetero_learnware.align(user_rkme, x_train, y_train)
            hetero_learnware_list.append(hetero_learnware)
        return hetero_learnware_list

    @staticmethod
    def single_aug_score(x_train, y_train, test_info):
        user_rkme, single_learnware = test_info["user_rkme"], test_info["single_learnware"]
        reuse_single_augment = HeteroMapAlignLearnware(single_learnware, mode="regression", **align_model_params)
        reuse_single_augment.align(user_rkme=user_rkme, x_train=x_train, y_train=y_train)
        return reuse_single_augment
    
    @staticmethod
    def multiple_aug_score(x_train, y_train, test_info):
        user_rkme, multiple_learnwares = test_info["user_rkme"], test_info["learnwares"]
        hetero_learnware_list = HeteroMethods.create_hetero_learnware_list(multiple_learnwares, user_rkme, x_train, y_train)
        reuse_multiple_augment = FeatureAugmentReuser(hetero_learnware_list, mode="regression")
        reuse_multiple_augment.fit(x_train=x_train, y_train=y_train)
        return reuse_multiple_augment
    
    @staticmethod
    def multiple_ensemble_pruning_score(x_train, y_train, test_info):
        user_rkme, multiple_learnwares = test_info["user_rkme"], test_info["learnwares"]
        hetero_learnware_list = HeteroMethods.create_hetero_learnware_list(multiple_learnwares, user_rkme, x_train, y_train)
        if len(hetero_learnware_list) == 1:
            return hetero_learnware_list[0]
        reuse_pruning = EnsemblePruningReuser(hetero_learnware_list, mode="regression")
        reuse_pruning.fit(val_X=x_train, val_y=y_train)
        return reuse_pruning
    
    @staticmethod
    def multiple_avg_score(x_train, y_train, test_info):
        user_rkme, multiple_learnwares = test_info["user_rkme"], test_info["learnwares"]
        hetero_learnware_list = HeteroMethods.create_hetero_learnware_list(multiple_learnwares, user_rkme, x_train, y_train)
        reuse_multiple_avg = AveragingReuser(hetero_learnware_list, mode="mean")
        return reuse_multiple_avg


test_methods = {
    "user_model": user_model_score,
    "hetero_single_aug": HeteroMethods.single_aug_score,
    "hetero_multiple_aug": HeteroMethods.multiple_aug_score,
    "hetero_multiple_avg": HeteroMethods.multiple_avg_score,
    "hetero_ensemble_pruning": HeteroMethods.multiple_ensemble_pruning_score,
    "homo_single_aug": HomoScoringMethods.single_aug_score,
    "homo_multiple_aug": HomoScoringMethods.multiple_aug_score,
    "homo_multiple_avg": HomoScoringMethods.multiple_avg_score,
    "homo_ensemble_pruning": HomoScoringMethods.multiple_ensemble_pruning_score
}