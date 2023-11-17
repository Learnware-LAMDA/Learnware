from __future__ import annotations

import codecs
import copy
import functools
import json
import os

from typing import Any

import numpy as np
import torch
import torch_optimizer
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from . import cnn_gp
from ..base import RegularStatSpecification
from ..table.rkme import rkme_solve_qp
from ....utils import choose_device, allocate_cuda_idx


class RKMEImageSpecification(RegularStatSpecification):
    # INNER_PRODUCT_COUNT = 0
    IMAGE_WIDTH = 32

    def __init__(self, cuda_idx: int = None, **kwargs):
        """Initializing RKME Image specification's parameters.

        Parameters
        ----------
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used. None indicates automatically choose device
        """
        self.RKME_IMAGE_VERSION = 1  # Please maintain backward compatibility.

        self.z = None
        self.beta = None
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self._device = choose_device(cuda_idx=self._cuda_idx)

        self.n_models = kwargs["n_models"] if "n_models" in kwargs else 16
        self.model_config = (
            {"k": 2, "mu": 0, "sigma": None, "net_width": 128, "net_depth": 3}
            if "model_config" not in kwargs
            else kwargs["model_config"]
        )

        super(RKMEImageSpecification, self).__init__(type=self.__class__.__name__)

    @property
    def device(self):
        return self._device

    def _generate_models(self, n_models: int, channel: int = 3, fixed_seed=None):
        model_class = functools.partial(_ConvNet_wide, channel=channel, **self.model_config)

        def __builder(i):
            if fixed_seed is not None:
                torch.manual_seed(fixed_seed[i])
            return model_class().to(self._device)

        return (__builder(m) for m in range(n_models))

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        K: int = 50,
        step_size: float = 0.01,
        steps: int = 100,
        resize: bool = True,
        nonnegative_beta: bool = True,
        reduce: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """Construct reduced set from raw dataset using iterative optimization.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in [N, C, H, W] format.
        K : int
            Size of the construced reduced set.
        step_size : float
            Step size for gradient descent in the iterative optimization.
        steps : int
            Total rounds in the iterative optimization.
        resize : bool
            Whether to scale the image to the requested size, by default True.
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        reduce : bool, optional
            Whether shrink original data to a smaller set, by default True
        verbose : bool, optional
            Whether to print training progress, by default True
        Returns
        -------

        """
        if len(X.shape) != 4:
            raise ValueError("X should be in shape of [N, C, H, W]. ")

        if (
            X.shape[2] != RKMEImageSpecification.IMAGE_WIDTH or X.shape[3] != RKMEImageSpecification.IMAGE_WIDTH
        ) and not resize:
            raise ValueError(
                "X should be in shape of [N, C, {0:d}, {0:d}]. "
                "Or set resize=True and the image will be automatically resized to {0:d} x {0:d}.".format(
                    RKMEImageSpecification.IMAGE_WIDTH
                )
            )

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).float()

        X[torch.isinf(X) | torch.isneginf(X) | torch.isposinf(X) | torch.isneginf(X)] = torch.nan
        if torch.any(torch.isnan(X)):
            for i, img in enumerate(X):
                is_nan = torch.isnan(img)
                if torch.any(is_nan):
                    if torch.all(is_nan):
                        raise ValueError(f"All values in image {i} are exceptional, e.g., NaN and Inf.")
                    img_mean = torch.nanmean(img)
                    X[i] = torch.where(is_nan, img_mean, img)

        if X.shape[2] != RKMEImageSpecification.IMAGE_WIDTH or X.shape[3] != RKMEImageSpecification.IMAGE_WIDTH:
            X = Resize((RKMEImageSpecification.IMAGE_WIDTH, RKMEImageSpecification.IMAGE_WIDTH), antialias=None)(X)

        num_points = X.shape[0]
        X_shape = X.shape
        Z_shape = tuple([K] + list(X_shape)[1:])

        X_train = (X - torch.mean(X, [0, 2, 3], keepdim=True)) / (torch.std(X, [0, 2, 3], keepdim=True))

        if X_train.shape[1] > 1 and ("whitening" not in kwargs or kwargs["whitening"]):
            whitening = _get_zca_matrix(X_train)
            X_train = X_train.reshape(num_points, -1) @ whitening
            X_train = X_train.view(*X_shape)

        if not reduce:
            self.beta = 1 / num_points * np.ones(num_points)
            self.z = torch.to(self._device)
            self.beta = torch.from_numpy(self.beta).to(self._device)
            return

        random_models = list(self._generate_models(n_models=self.n_models, channel=X.shape[1]))
        self.z = torch.zeros(Z_shape).to(self._device).float().normal_(0, 1)
        with torch.no_grad():
            x_features = self._generate_random_feature(X_train, random_models=random_models)
        self._update_beta(x_features, nonnegative_beta, random_models=random_models)

        optimizer = torch_optimizer.AdaBelief([{"params": [self.z]}], lr=step_size, eps=1e-16)

        for _ in tqdm(range(steps)) if verbose else range(steps):
            # Regenerate Random Models
            random_models = list(self._generate_models(n_models=self.n_models, channel=X.shape[1]))

            with torch.no_grad():
                x_features = self._generate_random_feature(X_train, random_models=random_models)
            self._update_z(x_features, optimizer, random_models=random_models)
            self._update_beta(x_features, nonnegative_beta, random_models=random_models)

    @torch.no_grad()
    def _update_beta(self, x_features: Any, nonnegative_beta: bool = True, random_models=None):
        Z = self.z
        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device)

        if not torch.is_tensor(x_features):
            x_features = torch.from_numpy(x_features)
        x_features = x_features.to(self._device)

        z_features = self._generate_random_feature(Z, random_models=random_models)
        K = self._calc_nngp_from_feature(z_features, z_features).to(self._device)
        C = self._calc_nngp_from_feature(z_features, x_features).to(self._device)
        C = torch.sum(C, dim=1) / x_features.shape[0]

        if nonnegative_beta:
            beta = rkme_solve_qp(K.double(), C.double())[0].to(self._device)
        else:
            beta = torch.linalg.inv(K + torch.eye(K.shape[0]).to(self._device) * 1e-5) @ C

        self.beta = beta

    def _update_z(self, x_features: Any, optimizer, random_models=None):
        Z = self.z
        beta = self.beta

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).float()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self._device)

        if not torch.is_tensor(x_features):
            x_features = torch.from_numpy(x_features)
        x_features = x_features.to(self._device).float()

        with torch.no_grad():
            beta = beta.unsqueeze(0)

        for i in range(3):
            z_features = self._generate_random_feature(Z, random_models=random_models)
            K_z = self._calc_nngp_from_feature(z_features, z_features)
            K_zx = self._calc_nngp_from_feature(x_features, z_features)
            term_1 = torch.sum(K_z * (beta.T @ beta))
            term_2 = torch.sum(K_zx * beta / x_features.shape[0])
            loss = term_1 - 2 * term_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _generate_random_feature(self, data_X, data_Y=None, batch_size=4096, random_models=None):
        X_features_list, Y_features_list = [], []

        dataset_X, dataset_Y = TensorDataset(data_X), None
        dataloader_X, dataloader_Y = DataLoader(dataset_X, batch_size=batch_size, shuffle=True), None
        if data_Y is not None:
            dataset_Y = TensorDataset(data_Y)
            dataloader_Y = DataLoader(dataset_Y, batch_size=batch_size, shuffle=True)
            assert data_X.shape[1] == data_Y.shape[1]

        for m, model in enumerate(
            random_models if random_models else self._generate_models(n_models=self.n_models, channel=data_X.shape[1])
        ):
            model.eval()

            curr_features_list = []
            for i, (X,) in enumerate(dataloader_X):
                out = model(X)
                curr_features_list.append(out)
            curr_features = torch.cat(curr_features_list, 0)
            X_features_list.append(curr_features)

            if data_Y is not None:
                curr_features_list = []
                for i, (Y,) in enumerate(dataloader_Y):
                    out = model(Y)
                    curr_features_list.append(out)
                curr_features = torch.cat(curr_features_list, 0)
                Y_features_list.append(curr_features)

        X_features = torch.cat(X_features_list, 1)
        X_features = X_features / torch.sqrt(torch.asarray(X_features.shape[1], device=self._device))
        if data_Y is None:
            return X_features
        else:
            Y_features = torch.cat(Y_features_list, 1)
            Y_features = Y_features / torch.sqrt(torch.asarray(Y_features.shape[1], device=self._device))
            return X_features, Y_features

    def inner_prod(self, Phi2: RKMEImageSpecification) -> float:
        """Compute the inner product between two RKME Image specifications

        Parameters
        ----------
        Phi2 : RKMEImageSpecification
            The other RKME Image specification.

        Returns
        -------
        float
            The inner product between two RKME Image specifications.
        """
        v = self._inner_prod_nngp(Phi2)
        return v

    def _inner_prod_nngp(self, Phi2: RKMEImageSpecification) -> float:
        beta_1 = self.beta.reshape(1, -1).detach().to(self._device)
        beta_2 = Phi2.beta.reshape(1, -1).detach().to(self._device)

        Z1 = self.z.to(self._device)
        Z2 = Phi2.z.to(self._device)

        kernel_fn = _build_ConvNet_NNGP(channel=Z1.shape[1], **self.model_config).to(self._device)
        if id(self) == id(Phi2):
            K_zz = kernel_fn(Z1)
        else:
            K_zz = kernel_fn(Z1, Z2)
        v = torch.sum(K_zz * (beta_1.T @ beta_2)).item()

        # RKMEImageSpecification.INNER_PRODUCT_COUNT += 1
        return v

    def dist(self, Phi2: RKMEImageSpecification, omit_term1: bool = False) -> float:
        """Compute the Maximum-Mean-Discrepancy(MMD) between two RKME Image specifications

        Parameters
        ----------
        Phi2 : RKMEImageSpecification
            The other RKME specification.
        omit_term1 : bool, optional
            True if the inner product of self with itself can be omitted, by default False.
        """

        if omit_term1:
            term1 = 0
        else:
            term1 = self.inner_prod(self)
        term2 = self.inner_prod(Phi2)
        term3 = Phi2.inner_prod(Phi2)

        v = float(term1 - 2 * term2 + term3)

        return v

    @staticmethod
    def _calc_nngp_from_feature(x1_feature: torch.Tensor, x2_feature: torch.Tensor):
        K_12 = x1_feature @ x2_feature.T + 0.01
        return K_12

    def herding(self, T: int) -> np.ndarray:
        raise NotImplementedError("The function herding hasn't been supported in Image RKME Specification.")

    def _sampling_candidates(self, N: int) -> np.ndarray:
        raise NotImplementedError()

    def get_beta(self) -> np.ndarray:
        return self.beta.detach().cpu().numpy()

    def get_z(self) -> np.ndarray:
        return self.z.detach().cpu().numpy()

    def save(self, filepath: str):
        """Save the computed RKME Image specification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path.
        """
        save_path = filepath
        rkme_to_save = self.get_states()
        if torch.is_tensor(rkme_to_save["z"]):
            rkme_to_save["z"] = rkme_to_save["z"].detach().cpu().numpy()
        rkme_to_save["z"] = rkme_to_save["z"].tolist()
        if torch.is_tensor(rkme_to_save["beta"]):
            rkme_to_save["beta"] = rkme_to_save["beta"].detach().cpu().numpy()
        rkme_to_save["beta"] = rkme_to_save["beta"].tolist()

        with codecs.open(save_path, "w", encoding="utf-8") as fout:
            json.dump(rkme_to_save, fout, separators=(",", ":"))

    def load(self, filepath: str) -> bool:
        """Load a RKME Image specification file in JSON format from the specified path.

        Parameters
        ----------
        filepath : str
            The specified loading path.

        Returns
        -------
        bool
            True if the RKME is loaded successfully.
        """
        # Load JSON file:
        load_path = filepath
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            rkme_load = json.loads(obj_text)
            rkme_load["z"] = torch.from_numpy(np.array(rkme_load["z"], dtype="float32"))
            rkme_load["beta"] = torch.from_numpy(np.array(rkme_load["beta"], dtype="float64"))

            for d in self.get_states():
                if d in rkme_load.keys():
                    setattr(self, d, rkme_load[d])

            self.beta = self.beta.to(self._device)
            self.z = self.z.to(self._device)

            return True
        else:
            return False


def _get_zca_matrix(X, reg_coef=0.1):
    X_flat = X.reshape(X.shape[0], -1)
    cov = (X_flat.T @ X_flat) / X_flat.shape[0]
    reg_amount = reg_coef * torch.trace(cov) / cov.shape[0]
    u, s, _ = torch.svd(cov + reg_amount * torch.eye(cov.shape[0]).to(X.device))
    inv_sqrt_zca_eigs = s ** (-0.5)
    whitening_transform = torch.einsum("ij,j,kj->ik", u, inv_sqrt_zca_eigs, u)

    return whitening_transform


class _ConvNet_wide(nn.Module):
    def __init__(self, channel, mu=None, sigma=None, k=2, net_width=128, net_depth=3, im_size=(32, 32)):
        self.k = k
        super().__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, im_size, mu, sigma)
        # self.aggregation = nn.AvgPool2d(kernel_size=shape_feat[1])

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        # out = self.aggregation(out).reshape(out.size(0), -1)
        return out

    def _make_layers(self, channel, net_width, net_depth, im_size, mu, sigma):
        k = self.k

        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [_build_conv2d_gaussian(in_channels, int(k * net_width), 3, 1, mean=mu, std=sigma)]
            shape_feat[0] = int(k * net_width)

            layers += [nn.ReLU(inplace=True)]
            in_channels = int(k * net_width)

            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            shape_feat[1] //= 2
            shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


def _build_conv2d_gaussian(in_channels, out_channels, kernel=3, padding=1, mean=None, std=None):
    layer = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    if mean is None:
        mean = 0
    if std is None:
        std = np.sqrt(2) / np.sqrt(layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3])
    # print('Initializing Conv. Mean=%.2f, std=%.2f'%(mean, std))
    torch.nn.init.normal_(layer.weight, mean, std)
    torch.nn.init.normal_(layer.bias, 0, 0.1)
    return layer


def _build_ConvNet_NNGP(channel, k=2, net_width=128, net_depth=3, kernel_size=3, im_size=(32, 32), **kwargs):
    layers = []
    for d in range(net_depth):
        layers += [cnn_gp.Conv2d(kernel_size=kernel_size, padding="same", var_bias=0.1, var_weight=np.sqrt(2))]
        # /np.sqrt(kernel_size * kernel_size * channel)
        layers += [cnn_gp.ReLU()]
        # AvgPooling
        layers += [cnn_gp.Conv2d(kernel_size=2, padding=0, stride=2)]

    assert im_size[0] % (2**net_depth) == 0
    layers.append(cnn_gp.Conv2d(kernel_size=im_size[0] // (2**net_depth), padding=0))

    return cnn_gp.Sequential(*layers)
