import torch
import random
import numpy as np
import geatpy as ea
from typing import List

from ..learnware import Learnware
from .base import BaseReuser
from ..logger import get_module_logger

logger = get_module_logger("ensemble_pruning")


class EnsemblePruningReuser(BaseReuser):
    """
    Baseline Multiple Learnware Reuser uing Marign Distribution guided multi-objective evolutionary Ensemble Pruning (MDEP) Method.

    References: [1] Yu-Chang Wu, Yi-Xiao He, Chao Qian, and Zhi-Hua Zhou. Multi-objective Evolutionary Ensemble Pruning Guided by Margin Distribution. In: Proceedings of the 17th International Conference on Parallel Problem Solving from Nature (PPSN'22), Dortmund, Germany, 2022.
    """

    def __init__(self, learnware_list: List[Learnware] = None, mode: str = "classification"):
        """The initialization method for ensemble pruning reuser

        Parameters
        ----------
        learnware_list : List[Learnware]
            The list contains learnwares
        mode : str
            - "regression" for regression task (learnware output is a real number)
            - "classification" for classification task (learnware output is a logitis vector or belongs to the set {0, 1, ..., class_num})
        """
        super(EnsemblePruningReuser, self).__init__(learnware_list)
        if mode not in ["regression", "classification"]:
            raise ValueError(f"Mode must be one of ['regression', 'classification'], but got {mode}")
        self.mode = mode
        self.selected_idxes = list(range(len(learnware_list)))

    def _MEDP_regression(self, v_predict: np.ndarray, v_true: np.ndarray, maxgen: int):
        """Selective ensemble for regression model

        Parameters
        ----------
        v_predict : np.ndarray
            - The output of models on validation set.
            - The dimension is (number of instances, number of models).
        v_true : np.ndarray
            - The ground truth of validation set.
            - The dimension is (number of instances, 1).
        maxgen : int
            - The maximum number of iteration rounds.

        Returns
        -------
        np.ndarray
            Binary one-dimensional vector, 1 indicates that the corresponding model is selected.
        """
        model_num = v_predict.shape[1]

        @ea.Problem.single
        def evalVars(Vars):
            while Vars.sum() <= 1:
                for i in range(0, model_num):
                    if random.random() < 1 / model_num:
                        Vars[i] = 1 if Vars[i] == 0 else 0

            vars_idxs = np.where(Vars == 1)[0].tolist()
            squared_diff = (v_predict[:, vars_idxs].mean(axis=1).reshape(-1, 1) - v_true) ** 2
            mse_loss = squared_diff.mean()

            f2 = [[mse_loss]]
            f3 = [[Vars.sum()]]
            ObjV = np.hstack([f2, f3])
            return ObjV

        npop = model_num
        Prophet = np.zeros((npop, model_num), dtype=np.int32)
        minf1, minf2, minf1forf2 = 1000, 1000, 1000
        minf1index, minf2index = 0, 0
        problem = ea.Problem(
            name="moea quick start",
            M=2,
            maxormins=[1, 1],
            Dim=model_num,
            varTypes=[1] * model_num,
            lb=[0] * model_num,
            ub=[1] * model_num,
            evalVars=evalVars,
        )

        for indi in range(0, model_num):
            Prophet[indi, indi] = 1
            objv = evalVars(Prophet[indi])
            if objv[0][0] < minf1 and objv[0][1] < minf1forf2:
                minf1 = objv[0][0]
                minf1index = indi
                minf1forf2 = objv[0][1]
            if objv[0][1] < minf2:
                minf2 = objv[0][1]
                minf2index = indi

        truePro = np.zeros((10, model_num), dtype=np.int32)
        truePro[0] = Prophet[minf1index]
        truePro[1] = Prophet[minf2index]
        for i in range(2, len(truePro)):
            truePro[i, random.randint(0, model_num - 1)] = 1

        # Choose MOEA such as: moea_NSGA3_templet  moea_MOEAD_templet to optimize.
        algorithm = ea.moea_NSGA2_templet(problem, ea.Population(Encoding="BG", NIND=npop), MAXGEN=maxgen, logTras=0)

        # Solve
        min_error_v = 100000
        res = ea.optimize(
            algorithm, verbose=True, drawing=0, outputMsg=False, drawLog=False, saveFlag=False, prophet=truePro
        )
        for pop in range(0, int(res["Vars"].size / model_num)):
            if min_error_v > res["ObjV"][pop][0]:
                min_error_v = res["ObjV"][pop][0]
                bst_pop = pop

        return res["Vars"][bst_pop]

    def _MEDP_multiclass(self, v_predict: np.ndarray, v_true: np.ndarray, maxgen: int):
        """Selective ensemble for multi-classification model

        Parameters
        ----------
        v_predict : np.ndarray
            - The output of models on validation set.
            - The dimension is (number of instances, number of models).
        v_true : np.ndarray
            - The ground truth of validation set.
            - The dimension is (number of instances, 1).
        maxgen : int
            - The maximum number of iteration rounds.

        Returns
        -------
        np.ndarray
            Binary one-dimensional vector, 1 indicates that the corresponding model is selected.
        """
        model_num = v_predict.shape[1]

        def find_top_two_freq(row):
            total = len(row)
            bincount = np.bincount(row)
            top1 = bincount.argmax()
            freq1 = bincount[top1]

            bincount[top1] = 0
            top2 = -1 if freq1 == total else bincount.argmax()
            freq2 = 0 if freq1 == total else bincount[top2]

            return top1, freq1, top2, freq2

        @ea.Problem.single
        def evalVars(Vars):
            while Vars.sum() <= 1:
                for i in range(0, model_num):
                    if random.random() < 1 / model_num:
                        Vars[i] = 1 if Vars[i] == 0 else 0

            # Extract the subscript whose vars value is 1
            idx = np.where(Vars == 1)[0]
            select = v_predict[:, idx]
            result = np.apply_along_axis(lambda x: find_top_two_freq(x), axis=1, arr=select)

            v_true_count = (select == v_true.reshape(-1, 1)).sum(axis=1)
            error_v = (result[:, 0] != v_true.reshape(-1)).sum()
            margin = result[:, 1] - result[:, 3]
            margin[result[:, 0] != v_true.reshape(-1)] = (v_true_count - result[:, 1])[
                result[:, 0] != v_true.reshape(-1)
            ]

            margin = margin / Vars.sum()
            mean_margin = np.mean(margin)
            f1 = [[100000]] if mean_margin <= 0 else [[np.std(margin) / (mean_margin)]]
            f2 = [[error_v]]
            f3 = [[Vars.sum()]]
            ObjV = np.hstack([f1, f2, f3])

            return ObjV

        npop = model_num
        Prophet = np.zeros((npop, model_num), dtype=np.int32)
        minf1, minf2, minf1forf2 = 1000, 1000, 1000
        minf1index, minf2index = 0, 0
        problem = ea.Problem(
            name="moea quick start",
            M=3,
            maxormins=[1, 1, 1],
            Dim=model_num,
            varTypes=[1] * model_num,
            lb=[0] * model_num,
            ub=[1] * model_num,
            evalVars=evalVars,
        )

        for indi in range(0, model_num):
            Prophet[indi, indi] = 1
            objv = evalVars(Prophet[indi])
            if objv[0][0] < minf1 and objv[0][1] < minf1forf2:
                minf1 = objv[0][0]
                minf1index = indi
                minf1forf2 = objv[0][1]
            if objv[0][1] < minf2:
                minf2 = objv[0][1]
                minf2index = indi

        truePro = np.zeros((10, model_num), dtype=np.int32)
        truePro[0] = Prophet[minf1index]
        truePro[1] = Prophet[minf2index]
        for i in range(2, len(truePro)):
            truePro[i, random.randint(0, model_num - 1)] = 1

        # Choose MOEA such as: moea_NSGA3_templet  moea_MOEAD_templet to optimize.
        algorithm = ea.moea_NSGA2_templet(problem, ea.Population(Encoding="BG", NIND=npop), MAXGEN=maxgen, logTras=0)

        # Solve
        min_erroe_v, choose_size, min_md = 100000, 100000, 100000
        res = ea.optimize(
            algorithm, verbose=True, drawing=0, outputMsg=False, drawLog=False, saveFlag=False, prophet=truePro
        )
        for pop in range(0, int(res["Vars"].size / model_num)):
            if min_erroe_v > res["ObjV"][pop][1]:
                min_erroe_v = res["ObjV"][pop][1]
                bst_pop = pop
                choose_size = res["ObjV"][pop][2]
                min_md = res["ObjV"][pop][0]

            if min_erroe_v == res["ObjV"][pop][1] and choose_size > res["ObjV"][pop][2]:
                choose_size = res["ObjV"][pop][2]
                bst_pop = pop

        return res["Vars"][bst_pop]

    def _MEDP_binaryclass(self, v_predict: np.ndarray, v_true: np.ndarray, maxgen: int):
        """Selective ensemble for binary classification model

        Parameters
        ----------
        v_predict : np.ndarray
            - The output of models on validation set.
            - The dimension is (number of instances, number of models).
        v_true : np.ndarray
            - The ground truth of validation set.
            - The dimension is (number of instances, 1).
        maxgen : int
            - The maximum number of iteration rounds.

        Returns
        -------
        np.ndarray
            Binary one-dimensional vector, 1 indicates that the corresponding model is selected.
        """
        model_num = v_predict.shape[1]
        v_predict[v_predict == 0.0] = -1
        v_true[v_true == 0.0] = -1

        @ea.Problem.single
        def evalVars(Vars):
            while Vars.sum() <= 1:
                for i in range(0, model_num):
                    if random.random() < 1 / model_num:
                        Vars[i] = 1 if Vars[i] == 0 else 0

            vars_idxs = np.where(Vars == 1)[0].tolist()
            margin = v_predict[:, vars_idxs].mean(axis=1).reshape(-1, 1) * v_true
            mean_margin = np.mean(margin)
            f1 = [[100000]] if mean_margin <= 0 else [[np.std(margin) / (mean_margin)]]
            error_v = (margin < 0).sum() + (margin == 0).sum() * 0.5

            f2 = [[error_v]]
            f3 = [[Vars.sum()]]
            ObjV = np.hstack([f1, f2, f3])

            return ObjV

        npop = model_num
        Prophet = np.zeros((npop, model_num), dtype=np.int32)
        minf1, minf2, minf1forf2 = 1000, 1000, 1000
        minf1index, minf2index = 0, 0
        problem = ea.Problem(
            name="moea quick start",
            M=3,
            maxormins=[1, 1, 1],
            Dim=model_num,
            varTypes=[1] * model_num,
            lb=[0] * model_num,
            ub=[1] * model_num,
            evalVars=evalVars,
        )

        for indi in range(0, model_num):
            Prophet[indi, indi] = 1
            objv = evalVars(Prophet[indi])
            if objv[0][0] < minf1 and objv[0][1] < minf1forf2:
                minf1 = objv[0][0]
                minf1index = indi
                minf1forf2 = objv[0][1]
            if objv[0][1] < minf2:
                minf2 = objv[0][1]
                minf2index = indi

        truePro = np.zeros((10, model_num), dtype=np.int32)
        truePro[0] = Prophet[minf1index]
        truePro[1] = Prophet[minf2index]
        for i in range(2, len(truePro)):
            truePro[i, random.randint(0, model_num - 1)] = 1

        # Choose MOEA such as: moea_NSGA3_templet  moea_MOEAD_templet to optimize.
        algorithm = ea.moea_NSGA3_templet(problem, ea.Population(Encoding="BG", NIND=npop), MAXGEN=maxgen, logTras=0)

        # Solve
        min_erroe_v, choose_size, min_md = 100000, 100000, 100000
        res = ea.optimize(
            algorithm, verbose=True, drawing=0, outputMsg=False, drawLog=False, saveFlag=False, prophet=truePro
        )
        for pop in range(0, int(res["Vars"].size / model_num)):
            if min_erroe_v > res["ObjV"][pop][1]:
                min_erroe_v = res["ObjV"][pop][1]
                bst_pop = pop
                choose_size = res["ObjV"][pop][2]
                min_md = res["ObjV"][pop][0]

            if min_erroe_v == res["ObjV"][pop][1] and choose_size > res["ObjV"][pop][2]:
                choose_size = res["ObjV"][pop][2]
                bst_pop = pop

        v_predict[v_predict == -1.0] = 0
        v_true[v_true == -1.0] = 0

        return res["Vars"][bst_pop]

    def _get_predict(self, X: np.ndarray, selected_idxes: List[int]):
        """Concatenate the output of learnwares corresponding to selected_idxes

        Parameters
        ----------
        X : np.ndarray
            Data that needs to be predicted
        selected_idxes : List[int]
            Learnware index list

        Returns
        -------
        np.ndarray
            Prediction given by each selected learnware
        """
        preds = []
        for idx in selected_idxes:
            pred_y = self.learnware_list[idx].predict(X)
            if isinstance(pred_y, torch.Tensor):
                pred_y = pred_y.detach().cpu().numpy()
            if not isinstance(pred_y, np.ndarray):
                raise TypeError(f"Model output must be np.ndarray or torch.Tensor")

            if len(pred_y.shape) == 1:
                pred_y = pred_y.reshape(-1, 1)
            elif len(pred_y.shape) == 2:
                if pred_y.shape[1] > 1:
                    pred_y = pred_y.argmax(axis=1).reshape(-1, 1)
            else:
                raise ValueError("Model output must be a 1D or 2D vector")
            preds.append(pred_y)

        return np.concatenate(preds, axis=1)

    def fit(self, val_X: np.ndarray, val_y: np.ndarray, maxgen: int = 500):
        """Ensemble pruning based on the validation set

        Parameters
        ----------
        val_X : np.ndarray
            Features of validation data.
        val_y : np.ndarray
            Labels of validation data.
        maxgen : int
            The maximum number of iteration rounds in ensemble pruning algorithms.
        """
        # Get the prediction of each learnware on the validation set
        v_predict = self._get_predict(val_X, list(range(len(self.learnware_list))))
        v_true = val_y.reshape(-1, 1)

        # Run ensemble pruning algorithm
        if self.mode == "regression":
            res = self._MEDP_regression(v_predict, v_true, maxgen)
        elif self.mode == "classification":
            if np.all((v_predict == 0) | (v_predict == 1)) and np.all((v_true == 0) | (v_true == 1)):
                res = self._MEDP_binaryclass(v_predict, v_true, maxgen)
            else:
                res = self._MEDP_multiclass(v_predict, v_true, maxgen)

        self.selected_idxes = np.where(res == 1)[0].tolist()

    def predict(self, user_data: np.ndarray) -> np.ndarray:
        """Prediction for user data using the final pruned ensemble

        Parameters
        ----------
        user_data : np.ndarray
            Raw user data.

        Returns
        -------
        np.ndarray
            Prediction given by ensemble method
        """
        preds = self._get_predict(user_data, self.selected_idxes)

        if self.mode == "regression":
            return preds.mean(axis=1)
        elif self.mode == "classification":
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)
