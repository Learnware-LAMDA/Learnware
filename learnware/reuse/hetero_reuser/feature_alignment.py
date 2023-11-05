from typing import List, Any
import numpy as np
from numpy import ndarray
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from tqdm import trange
from loguru import logger

from learnware.learnware import Learnware
from learnware.specification import RKMEStatSpecification
from learnware.specification.regular.table.rkme import choose_device

from ..base import BaseReuser


class FeatureAligner(BaseReuser):

    def __init__(self, learnware: Learnware = None, task_type: str = None, cuda_idx=0, **align_arguments):
        self.learnware=learnware
        assert task_type in ["classification", "regression"]
        self.task_type=task_type
        self.align_arguments=align_arguments
        self.cuda_idx=cuda_idx
        self.device = choose_device(cuda_idx=cuda_idx)

    def fit(self, user_rkme):
        target_rkme=self.learnware.specification.get_stat_spec()["RKMEStatSpecification"]
        trainer=FeatureAlignmentTrainer(target_rkme=target_rkme, user_rkme=user_rkme, cuda_idx=self.cuda_idx, **self.align_arguments)
        self.align_model=trainer.model
        self.align_model.eval()

    def predict(self, user_data: ndarray) -> ndarray:
        user_data=self._fill_data(user_data)
        transformed_user_data=self.align_model(torch.tensor(user_data, device=self.device).float()).detach().cpu().numpy()
        y_pred=self.learnware.predict(transformed_user_data)
        return y_pred
    
    def _fill_data(self, X: np.ndarray):
        X[np.isinf(X) | np.isneginf(X) | np.isposinf(X) | np.isneginf(X)] = np.nan
        if np.any(np.isnan(X)):
            for col in range(X.shape[1]):
                is_nan = np.isnan(X[:, col])
                if np.any(is_nan):
                    if np.all(is_nan):
                        raise ValueError(f"All values in column {col} are exceptional, e.g., NaN and Inf.")
                    # Fill np.nan with np.nanmean
                    col_mean = np.nanmean(X[:, col])
                    X[:, col] = np.where(is_nan, col_mean, X[:, col])
        return X


class FeatureAlignmentModel(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[1024], activation="relu", dropout_ratio=0, use_bn=False):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.fc_list = nn.ModuleList()
        self.drop_list = nn.ModuleList()

        if len(hidden_dims) > 0:
            for i in range(len(dims) - 2):
                self.drop_list.append(nn.Dropout(dropout_ratio))
                if use_bn:
                    self.fc_list.append(nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1])))
                else:
                    self.fc_list.append(nn.Linear(dims[i], dims[i + 1]))

        self.final_fc = nn.Linear(dims[-2], dims[-1])

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "selu":
            self.activation = F.selu
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

    def forward(self, x):
        if len(self.fc_list) > 0:
            for fc, drop in zip(self.fc_list, self.drop_list):
                x = fc(x)
                x = self.activation(x)
                x = drop(x)
        return self.final_fc(x)
    

class FeatureAlignmentTrainer():

    def __init__(
        self,
        target_rkme: RKMEStatSpecification,  # (X, weight)
        user_rkme: RKMEStatSpecification,  # (X, weight)
        extra_labeled_data: Any = None,
        target_learnware: Learnware = None,
        num_epoch: int = 50,
        lr: float = 1e-3,
        gamma: float = 0.1,
        network_type: str = "ArbitraryMapping",
        optimizer_type: str = "Adam",
        hidden_dims: List[int] = [1024],
        activation: str = "relu",
        dropout_ratio: float = 0,
        use_bn: bool = False,
        const: float = 1e1,
        cuda_idx: int = 0
    ):
        """Training the base mapping network
        """
        self.target_rkme = target_rkme
        self.user_rkme = user_rkme
        self.args = {
            "lr": lr,
            "num_epoch": num_epoch,
            "gamma": gamma,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout_ratio": dropout_ratio,
            "use_bn": use_bn,
        }
        self.network_type = network_type
        self.optimizer_type = optimizer_type
        self.const=const
        self.device = choose_device(cuda_idx=cuda_idx)
        if extra_labeled_data is not None and target_learnware is not None:
            self.train_with_labeled_data(extra_labeled_data[0], extra_labeled_data[1], target_learnware)
        else:
            self.train()

    def gaussian_kernel(self, x1, x2):
        x1 = x1.double()
        x2 = x2.double()
        X12norm = torch.sum(x1**2, 1, keepdim=True) - 2 * x1 @ x2.T + torch.sum(x2**2, 1, keepdim=True).T
        return torch.exp(-X12norm * self.args["gamma"])

    def compute_mmd(self, user_X, user_weight, target_X, target_weight):
        term1 = torch.sum(self.gaussian_kernel(user_X, user_X) * (user_weight.T @ user_weight))
        term2 = torch.sum(self.gaussian_kernel(user_X, target_X) * (user_weight.T @ target_weight))
        term3 = torch.sum(self.gaussian_kernel(target_X, target_X) * (target_weight.T @ target_weight))
        return term1 - 2 * term2 + term3

    def train(self):
        args = self.args
        input_dim = self.user_rkme.get_z().shape[1]
        output_dim = self.target_rkme.get_z().shape[1]

        user_model=FeatureAlignmentModel(input_dim, output_dim, args["hidden_dims"], args["activation"], args["dropout_ratio"], args["use_bn"])

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_model.to(self.device)
        user_data_x = torch.tensor(self.user_rkme.get_z(), device=self.device).float()
        user_data_weight = torch.tensor(self.user_rkme.get_beta(), device=self.device).view(1, -1).double()
        target_data_x = torch.tensor(self.target_rkme.get_z(), device=self.device)
        target_data_weight = torch.tensor(self.target_rkme.get_beta(), device=self.device).view(1, -1).double()
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(user_model.parameters(), lr=args["lr"])
        else:
            optimizer = torch.optim.SGD(user_model.parameters(), lr=args["lr"])

        start_time = time.time()
        for epoch in trange(args["num_epoch"], desc="Epoch"):
            transformed_user_data_x = user_model(user_data_x)
            mmd_loss = self.compute_mmd(transformed_user_data_x, user_data_weight, target_data_x, target_data_weight)

            optimizer.zero_grad()
            mmd_loss.backward()
            optimizer.step()
            logger.info(
                "epoch: {}, train mmd_loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs".format(
                    epoch, mmd_loss.item(), optimizer.param_groups[0]["lr"], time.time() - start_time
                )
            )

        self.model = user_model
        logger.info("training complete, cost {:.1f} secs.".format(time.time() - start_time))