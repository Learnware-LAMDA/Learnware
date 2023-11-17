import time
import torch
import numpy as np
import torch.nn as nn
from typing import List
from tqdm import trange
import torch.nn.functional as F

from ..align import AlignLearnware
from ..utils import fill_data_with_mean
from ...utils import choose_device, allocate_cuda_idx
from ...logger import get_module_logger
from ...learnware import Learnware
from ...specification import RKMETableSpecification

logger = get_module_logger("feature_align")


class FeatureAlignLearnware(AlignLearnware):
    """
    FeatureAlignLearnware is a class for aligning features from a user dataset with a target dataset using a learnware model.
    It supports both classification and regression tasks and uses a feature alignment trainer for alignment.

    Attributes
    ----------
    learnware : Learnware
        The learnware model used for final prediction.
    align_arguments : dict
        Additional arguments for the feature alignment trainer.
    cuda_idx : int
        Index of the CUDA device to be used for computations.
    """

    def __init__(self, learnware: Learnware, cuda_idx=None, **align_arguments):
        """
        Initialize the FeatureAlignLearnware with a learnware model, mode, CUDA device index, and alignment arguments.

        Parameters
        ----------
        learnware : Learnware
            A learnware model used for initial predictions.
        cuda_idx : int
            The index of the CUDA device for computations.
        align_arguments : dict
            Additional arguments to be passed to the feature alignment trainer.
        """
        super(FeatureAlignLearnware, self).__init__(learnware)
        self.align_arguments = align_arguments
        self.cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self.device = choose_device(cuda_idx=self.cuda_idx)
        self.align_model = None

    def align(self, user_rkme: RKMETableSpecification):
        """
        Train the align model using the RKME specifications from the user and the learnware.

        Parameters
        ----------
        user_rkme : RKMETableSpecification
            The RKME specification from the user dataset.
        """
        target_rkme = self.specification.get_stat_spec()["RKMETableSpecification"]
        trainer = FeatureAlignTrainer(
            target_rkme=target_rkme, user_rkme=user_rkme, cuda_idx=self.cuda_idx, **self.align_arguments
        )
        self.align_model = trainer.model
        self.align_model.eval()

    def predict(self, user_data: np.ndarray) -> np.ndarray:
        """
        Predict the output for user data using the aligned model and learnware model.

        Parameters
        ----------
        user_data : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            Predicted output from the learnware model after alignment.
        """
        assert self.align_model is not None, "FeatureAlignLearnware must be aligned before making predictions."
        user_data = fill_data_with_mean(user_data)
        transformed_user_data = (
            self.align_model(torch.tensor(user_data, device=self.device).float()).detach().cpu().numpy()
        )
        y_pred = super(FeatureAlignLearnware, self).predict(transformed_user_data)
        return y_pred


class FeatureAlignModel(nn.Module):
    """
    FeatureAlignModel is a neural network module designed for feature alignment tasks.
    It consists of multiple fully connected (dense) layers, optional dropout and batch normalization layers,
    and supports different activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [1024],
        activation: str = "relu",
        dropout_ratio: float = 0,
        use_bn: bool = False,
    ):
        """
        Initialize the FeatureAlignModel.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the input features.
        output_dim : int
            The dimensionality of the output features.
        hidden_dims : List[int], optional
            A list specifying the number of units in each hidden layer.
        activation : str, optional
            The activation function to use. Supported options are "relu", "gelu", "selu", and "leakyrelu".
        dropout_ratio : float, optional
            The dropout ratio applied to each layer (0 means no dropout).
        use_bn : bool, optional
            Whether to use batch normalization after each fully connected layer.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.
        """
        if len(self.fc_list) > 0:
            for fc, drop in zip(self.fc_list, self.drop_list):
                x = fc(x)  # Apply fully connected layer
                x = self.activation(x)  # Apply activation function
                x = drop(x)  # Apply dropout

        # Return output from final fully connected layer
        return self.final_fc(x)


class FeatureAlignTrainer:
    """
    FeatureAlignTrainer is a class designed to train a neural network for aligning features from a user dataset
    to a target dataset. It utilizes Maximum Mean Discrepancy (MMD) as the loss function for training.

    Attributes
    ----------
    target_rkme : RKMETableSpecification
        The RKME specification of the target dataset.
    user_rkme : RKMETableSpecification
        The RKME specification of the user dataset.
    num_epoch : int
        The number of training epochs.
    lr : float
        Learning rate for the optimizer.
    gamma : float
        The gamma parameter for the Gaussian kernel in MMD computation.
    network_type : str
        Type of the neural network used for feature alignment.
    optimizer_type : str
        Type of optimizer to be used in training ('Adam' or 'SGD').
    hidden_dims : List[int]
        A list specifying the number of units in each hidden layer.
    activation : str
        The activation function to use in the network.
    dropout_ratio : float
        The dropout ratio applied to each layer.
    use_bn : bool
        Whether to use batch normalization after each fully connected layer.
    const : float
        A constant value used in training.
    cuda_idx : int
        Index of the CUDA device to be used for computations.
    """

    def __init__(
        self,
        target_rkme: RKMETableSpecification,
        user_rkme: RKMETableSpecification,
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
        cuda_idx: int = None,
    ):
        """
        Initialize the FeatureAlignTrainer with the specified parameters.
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
        self.const = const
        self.cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self.device = choose_device(cuda_idx=self.cuda_idx)
        self.train()

    def gaussian_kernel(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute the Gaussian kernel between two sets of samples.

        Parameters
        ----------
        x1 : torch.Tensor
            First set of samples.
        x2 : torch.Tensor
            Second set of samples.

        Returns
        -------
        torch.Tensor
            The computed Gaussian kernel matrix.
        """
        x1 = x1.double()
        x2 = x2.double()
        X12norm = torch.sum(x1**2, 1, keepdim=True) - 2 * x1 @ x2.T + torch.sum(x2**2, 1, keepdim=True).T
        return torch.exp(-X12norm * self.args["gamma"])

    def compute_mmd(
        self, user_X: torch.Tensor, user_weight: torch.Tensor, target_X: torch.Tensor, target_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between the user and target datasets.

        Parameters
        ----------
        user_X : torch.Tensor
            Transformed user data.
        user_weight : torch.Tensor
            Weights of the user data.
        target_X : torch.Tensor
            Target data.
        target_weight : torch.Tensor
            Weights of the target data.

        Returns
        -------
        torch.Tensor
            The computed MMD loss.
        """
        term1 = torch.sum(self.gaussian_kernel(user_X, user_X) * (user_weight.T @ user_weight))
        term2 = torch.sum(self.gaussian_kernel(user_X, target_X) * (user_weight.T @ target_weight))
        term3 = torch.sum(self.gaussian_kernel(target_X, target_X) * (target_weight.T @ target_weight))
        return term1 - 2 * term2 + term3

    def train(self):
        """
        Train the feature alignment model using MMD as the loss function.
        """
        args = self.args
        input_dim = self.user_rkme.get_z().shape[1]
        output_dim = self.target_rkme.get_z().shape[1]

        user_model = FeatureAlignModel(
            input_dim, output_dim, args["hidden_dims"], args["activation"], args["dropout_ratio"], args["use_bn"]
        )
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
