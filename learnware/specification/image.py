from __future__ import annotations

import codecs
import copy
import functools
import json
import os

from typing import Any, Union

import numpy as np
import torch
import torch_optimizer
from torch import nn
from torch.func import jacrev, functional_call
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from .base import BaseStatSpecification
from .rkme import solve_qp, choose_device, setup_seed


class RKMEImageStatSpecification(BaseStatSpecification):
    inner_prod_buffer = dict()
    INNER_PRODUCT_COUNT = 0
    IMAGE_WIDTH = 32

    def __init__(self, cuda_idx: int = -1, buffering: bool=True, **kwargs):
        """Initializing RKME Image specification's parameters.

        Parameters
        ----------
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used.
        buffering: bool
            When buffering is True, the result of inner_prod will be buffered according to id(object), avoiding duplicate kernel function calculations, by default True.
        """
        self.RKME_IMAGE_VERSION = 1 # Please maintain backward compatibility.
        # torch.cuda.empty_cache()

        self.z = None
        self.beta = None
        self.cuda_idx = cuda_idx
        self.device = choose_device(cuda_idx=cuda_idx)
        self.buffering = buffering

        self.n_models = kwargs["n_models"] if "n_models" in kwargs else 16
        self.model_config = {
            "k": 2, "mu": 0, "sigma": None, 'chopped_head': True,
            "net_width": 128, "net_depth": 3, "net_act": "relu"
        } if "model_config" not in kwargs else kwargs["model_config"]

        setup_seed(0)

    def _generate_models(self, n_models: int, channel: int=3, fixed_seed=None):
        model_class = functools.partial(_ConvNet_wide, channel=channel, **self.model_config)

        def __builder(i):
            if fixed_seed is not None:
                torch.manual_seed(fixed_seed[i])
            return model_class().to(self.device)

        return (__builder(m) for m in range(n_models))

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        K: int = 50,
        step_size: float = 0.01,
        steps: int=100,
        resize: bool = False,
        nonnegative_beta: bool = True,
        reduce: bool = True
    ):
        """Construct reduced set from raw dataset using iterative optimization.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format.
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

        Returns
        -------

        """
        if  (X.shape[2] != RKMEImageStatSpecification.IMAGE_WIDTH or
             X.shape[3] != RKMEImageStatSpecification.IMAGE_WIDTH) and not resize:
            raise ValueError("X should be in shape of [N, C, {0:d}, {0:d}]. "
                             "Or set resize=True and the image will be automatically resized to {0:d} x {0:d}."
                             .format(RKMEImageStatSpecification.IMAGE_WIDTH))

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device)

        X[torch.isinf(X) | torch.isneginf(X) | torch.isposinf(X) | torch.isneginf(X)] = torch.nan
        if torch.any(torch.isnan(X)):
            for i, img in enumerate(X):
                is_nan = torch.isnan(img)
                if torch.any(is_nan):
                    if torch.all(is_nan):
                        raise ValueError(f"All values in image {i} are exceptional, e.g., NaN and Inf.")
                    img_mean = torch.nanmean(img)
                    X[i] = torch.where(is_nan, img_mean, img)

        if (X.shape[2] != RKMEImageStatSpecification.IMAGE_WIDTH or
             X.shape[3] != RKMEImageStatSpecification.IMAGE_WIDTH):
            X = Resize((RKMEImageStatSpecification.IMAGE_WIDTH,
                        RKMEImageStatSpecification.IMAGE_WIDTH))(X)

        num_points = X.shape[0]
        X_shape = X.shape
        Z_shape = tuple([K] + list(X_shape)[1:])

        X_train = (X - torch.mean(X, [0, 2, 3], keepdim=True)) / (torch.std(X, [0, 2, 3], keepdim=True))
        if X_train.shape[1] > 1:
            whitening = _get_zca_matrix(X_train)
            X_train = X_train.reshape(num_points, -1) @ whitening
            X_train = X_train.view(*X_shape)

        if not reduce:
            self.beta = 1 / num_points * np.ones(num_points)
            self.z = torch.to(self.device)
            self.beta = torch.from_numpy(self.beta).to(self.device)
            return

        random_models = list(self._generate_models(n_models=self.n_models, channel=X.shape[1]))
        self.z = torch.zeros(Z_shape).to(self.device).float().normal_(0, 1)
        with torch.no_grad():
            x_features = self._generate_random_feature(X_train, random_models=random_models)
        self._update_beta(x_features, nonnegative_beta, random_models=random_models)

        optimizer = torch_optimizer.AdaBelief([{"params": [self.z]}],
                                              lr=step_size, eps=1e-16)

        for i in tqdm(range(steps), total=steps):
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
        Z = Z.to(self.device)

        if not torch.is_tensor(x_features):
            x_features = torch.from_numpy(x_features)
        x_features = x_features.to(self.device)

        z_features = self._generate_random_feature(Z, random_models=random_models)
        K = self._calc_ntk_from_feature(z_features, z_features).to(self.device)
        C = self._calc_ntk_from_feature(z_features, x_features).to(self.device)
        C = torch.sum(C, dim=1) / x_features.shape[0]

        if nonnegative_beta:
            beta = solve_qp(K.double(), C.double()).to(self.device)
        else:
            beta = torch.linalg.inv(K + torch.eye(K.shape[0]).to(self.device) * 1e-5) @ C

        self.beta = beta

    def _update_z(self, x_features: Any, optimizer, random_models=None):
        Z = self.z
        beta = self.beta

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self.device).float()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self.device)

        if not torch.is_tensor(x_features):
            x_features = torch.from_numpy(x_features)
        x_features = x_features.to(self.device).float()

        with torch.no_grad():
            beta = beta.unsqueeze(0)

        for i in range(3):
            z_features = self._generate_random_feature(Z, random_models=random_models)
            K_z = self._calc_ntk_from_feature(z_features, z_features)
            K_zx = self._calc_ntk_from_feature(x_features, z_features)
            term_1 = torch.sum(K_z * (beta.T @ beta))
            term_2 = torch.sum(K_zx * beta / x_features.shape[0])
            loss = term_1 - 2 * term_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _generate_random_feature(self, data_X, batch_size=4096, random_models=None) -> torch.Tensor:
        X_features_list = []
        if not torch.is_tensor(data_X):
            data_X = torch.from_numpy(data_X)
        data_X = data_X.to(self.device)

        dataset = TensorDataset(data_X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for m, model in enumerate(random_models if random_models else
                                  self._generate_models(n_models=self.n_models, channel=data_X.shape[1])):
            model.eval()
            curr_features_list = []
            for i, (X,) in enumerate(dataloader):
                out = model(X)
                curr_features_list.append(out)
            curr_features = torch.cat(curr_features_list, 0)
            X_features_list.append(curr_features)
        X_features = torch.cat(X_features_list, 1)
        X_features = X_features / torch.sqrt(torch.asarray(X_features.shape[1], device=self.device))

        return X_features

    def inner_prod(self, Phi2: RKMEImageStatSpecification) -> float:
        """Compute the inner product between two RKME Image specifications

        Parameters
        ----------
        Phi2 : RKMEImageStatSpecification
            The other RKME Image specification.

        Returns
        -------
        float
            The inner product between two RKME Image specifications.
        """

        if self.buffering and Phi2.buffering:
            if (id(self), id(Phi2)) in RKMEImageStatSpecification.inner_prod_buffer:
                return RKMEImageStatSpecification.inner_prod_buffer[(id(self), id(Phi2))]

        v = self._inner_prod_ntk(Phi2)
        if self.buffering and Phi2.buffering:
            RKMEImageStatSpecification.inner_prod_buffer[(id(self), id(Phi2))] = v
            RKMEImageStatSpecification.inner_prod_buffer[(id(Phi2), id(self))] = v
        return v

    def _inner_prod_ntk(self, Phi2: RKMEImageStatSpecification) -> float:
        beta_1 = self.beta.reshape(1, -1).detach()
        beta_2 = Phi2.beta.reshape(1, -1).detach()

        Z1 = self.z.to(self.device)
        Z2 = Phi2.z.to(self.device)

        # Use the old way
        assert Z1.shape[1] == Z2.shape[1]
        random_models = list(self._generate_models(n_models=self.n_models * 4, channel=Z1.shape[1]))
        z1_features = self._generate_random_feature(Z1, random_models=random_models)
        z2_features = self._generate_random_feature(Z2, random_models=random_models)
        K_zz = self._calc_ntk_from_feature(z1_features, z2_features)

        v = torch.sum(K_zz * (beta_1.T @ beta_2)).item()

        RKMEImageStatSpecification.INNER_PRODUCT_COUNT += 1
        return v

    def dist(self, Phi2: RKMEImageStatSpecification, omit_term1: bool = False) -> float:
        """Compute the Maximum-Mean-Discrepancy(MMD) between two RKME Image specifications

        Parameters
        ----------
        Phi2 : RKMEImageStatSpecification
            The other RKME specification.
        omit_term1 : bool, optional
            True if the inner product of self with itself can be omitted, by default False.
        """

        with torch.no_grad():
            if omit_term1:
                term1 = 0
            else:
                term1 = self.inner_prod(self)
            term2 = self.inner_prod(Phi2)
            term3 = Phi2.inner_prod(Phi2)

        return float(term1 - 2 * term2 + term3)

    @staticmethod
    def _calc_ntk_from_feature(x1_feature: torch.Tensor, x2_feature: torch.Tensor):
        K_12 = x1_feature @ x2_feature.T + 0.01
        return K_12

    def _calc_ntk_empirical(self, x1: torch.Tensor, x2: torch.Tensor):
        if x1.shape[1] != x2.shape[1]:
            raise ValueError("The channel of two rkme image specification should be equal (e.g. 3 or 1).")

        results = []
        for m, model in enumerate(self._generate_models(n_models=self.n_models, channel=x1.shape[1])):
            # Compute J(x1)
            # jac1 = vamp(lambda x: jacrev(lambda p, i: functional_call(model, p, i), argnums=0)(dict(model.named_parameters()), x))(x1)
            jac1 = jacrev(lambda p, i: functional_call(model, p, i), argnums=0)(dict(model.named_parameters()), x1)
            jac1 = [j.flatten(2) for j in jac1]

            # Compute J(x2)
            jac2 = functional_call(model, model.parameters(), x2)
            jac2 = [j.flatten(2) for j in jac2]

            result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
            results.append(result.sum(0))

        results = torch.stack(results)
        return results.mean(0)

    def herding(self, T: int) -> np.ndarray:
        raise NotImplementedError(
            "The function herding hasn't been supported in Image RKME Specification.")

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
        rkme_to_save = copy.deepcopy(self.__dict__)
        if torch.is_tensor(rkme_to_save["z"]):
            rkme_to_save["z"] = rkme_to_save["z"].detach().cpu().numpy()
        rkme_to_save["z"] = rkme_to_save["z"].tolist()
        if torch.is_tensor(rkme_to_save["beta"]):
            rkme_to_save["beta"] = rkme_to_save["beta"].detach().cpu().numpy()
        rkme_to_save["beta"] = rkme_to_save["beta"].tolist()
        rkme_to_save["device"] = "gpu" if rkme_to_save["cuda_idx"] != -1 else "cpu"

        json.dump(
            rkme_to_save,
            codecs.open(save_path, "w", encoding="utf-8"),
            separators=(",", ":"),
        )

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
            rkme_load["device"] = choose_device(rkme_load["cuda_idx"])
            rkme_load["z"] = torch.from_numpy(np.array(rkme_load["z"])).float()
            rkme_load["beta"] = torch.from_numpy(np.array(rkme_load["beta"]))

            for d in self.__dir__():
                if d in rkme_load.keys():
                    setattr(self, d, rkme_load[d])

            self.beta = self.beta.to(self.device)
            self.z = self.z.to(self.device)

            return True
        else:
            return False


def _get_zca_matrix(X, reg_coef=0.1):
    X_flat = X.reshape(X.shape[0], -1)
    cov = (X_flat.T @ X_flat) / X_flat.shape[0]
    reg_amount = reg_coef * torch.trace(cov) / cov.shape[0]
    u, s, _ = torch.svd(cov.cuda() + reg_amount * torch.eye(cov.shape[0]).cuda())
    inv_sqrt_zca_eigs = s ** (-0.5)
    whitening_transform = torch.einsum(
        'ij,j,kj->ik', u, inv_sqrt_zca_eigs, u)

    return whitening_transform


class _ConvNet_wide(nn.Module):
    def __init__(self, channel, mu=None, sigma=None, k=4, net_width=128, net_depth=3,
                 net_act='relu', net_norm='none', net_pooling='avgpooling', im_size=(32, 32), chopped_head=False):
        self.k = k
        # print('Building Conv Model')
        super().__init__()

        # net_depth = 1
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm,
                                                      net_act, net_pooling, im_size, mu, sigma)
        # print(shape_feat)
        self.chopped_head = chopped_head

    def forward(self, x):
        out = self.features(x)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        # print(out.size())
        return out

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, mu, sigma):
        k = self.k

        layers = []
        in_channels = channel
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [build_conv2d_gaussian(in_channels, int(k * net_width), 3,
                                             1, mean=mu, std=sigma)]
            shape_feat[0] = int(k * net_width)

            layers += [nn.ReLU(inplace=True)]
            in_channels = int(k * net_width)

            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            shape_feat[1] //= 2
            shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

def build_conv2d_gaussian(in_channels, out_channels, kernel=3, padding=1, mean=None, std=None):
    layer = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    if mean is None:
        mean = 0
    if std is None:
        std = np.sqrt(2)/np.sqrt(layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3])
    # print('Initializing Conv. Mean=%.2f, std=%.2f'%(mean, std))
    torch.nn.init.normal_(layer.weight, mean, std)
    torch.nn.init.normal_(layer.bias, 0, .1)
    return layer