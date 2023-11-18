from __future__ import annotations

import os
import copy
import torch
import json
import codecs
import random
import numpy as np
from qpsolvers import solve_qp, Problem, solve_problem
from collections import Counter
from typing import Tuple, Any, List, Union, Dict
import scipy
from sklearn.cluster import MiniBatchKMeans

from ..base import RegularStatSpecification
from ....logger import get_module_logger
from ....utils import allocate_cuda_idx, choose_device

logger = get_module_logger("rkme")


class RKMETableSpecification(RegularStatSpecification):
    """Reduced Kernel Mean Embedding (RKME) Specification"""

    def __init__(self, gamma: float = 0.1, cuda_idx: int = None):
        """Initializing RKME parameters.

        Parameters
        ----------
        gamma : float
            Bandwidth in gaussian kernel, by default 0.1.
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used. None indicates automatically choose device
        """
        self.z = None
        self.beta = None
        self.gamma = gamma
        self.num_points = 0
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        torch.cuda.empty_cache()
        self._device = choose_device(cuda_idx=self._cuda_idx)
        super(RKMETableSpecification, self).__init__(type=self.__class__.__name__)

    @property
    def device(self):
        return self._device

    def get_beta(self) -> np.ndarray:
        """Move beta(RKME weights) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of beta in CPU memory.
        """
        return self.beta.detach().cpu().numpy()

    def get_z(self) -> np.ndarray:
        """Move z(RKME reduced set points) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of z in CPU memory.
        """
        return self.z.detach().cpu().numpy()

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        K: int = 100,
        step_size: float = 0.1,
        steps: int = 3,
        nonnegative_beta: bool = True,
        reduce: bool = True,
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
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        reduce : bool, optional
            Whether shrink original data to a smaller set, by default True
        """
        alpha = None
        self.num_points = X.shape[0]
        X_shape = X.shape
        Z_shape = tuple([K] + list(X_shape)[1:])
        X = X.reshape(self.num_points, -1)

        # Check data values
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

        if not reduce:
            self.z = X.reshape(X_shape)
            self.beta = 1 / self.num_points * np.ones(self.num_points)
            self.z = torch.from_numpy(self.z).double().to(self._device)
            self.beta = torch.from_numpy(self.beta).double().to(self._device)
            return

        # Initialize Z by clustering, utiliing kmeans to speed up the process.
        self._init_z_by_kmeans(X, K)
        self._update_beta(X, nonnegative_beta)

        # Alternating optimize Z and beta
        for i in range(steps):
            self._update_z(alpha, X, step_size)
            self._update_beta(X, nonnegative_beta)

        # Reshape to original dimensions
        self.z = self.z.reshape(Z_shape)

    def _init_z_by_kmeans(self, X: Union[np.ndarray, torch.tensor], K: int):
        """Intialize Z by kmeans clustering.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        K : int
            Size of the construced reduced set.
        """
        X = X.astype("float32")
        kmeans = MiniBatchKMeans(n_clusters=K, max_iter=100, verbose=False, n_init="auto")
        kmeans.fit(X)
        center = torch.from_numpy(kmeans.cluster_centers_).double()
        self.z = center

    def _update_beta(self, X: Any, nonnegative_beta: bool = True):
        """Fix Z and update beta using its closed-form solution.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        """
        Z = self.z
        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).double()
        K = torch_rbf_kernel(Z, Z, gamma=self.gamma).to(self._device)
        C = torch_rbf_kernel(Z, X, gamma=self.gamma).to(self._device)
        C = torch.sum(C, dim=1) / X.shape[0]

        if nonnegative_beta:
            beta, _ = rkme_solve_qp(K, C)
            beta = beta.to(self._device)
        else:
            beta = torch.linalg.inv(K + torch.eye(K.shape[0]).to(self._device) * 1e-5) @ C

        self.beta = beta

    @torch.no_grad()
    def _update_z_vectorize(self, alpha: float, X: Any, step_size: float, batch_size=8):
        """Fix beta and update Z using gradient descent.
        Unlike method _update_z, this method updates z by batches.

        Parameters
        ----------
        alpha : int
            Normalization factor.
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        step_size : float
            Step size for gradient descent.
        batch_size : int
            To prevent exceeding GPU memory, process no more than batch_size at a time.
        """
        gamma = self.gamma
        Z = self.z
        beta = self.beta

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self._device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).double()

        grad_Z = torch.zeros_like(Z)
        for i in range(0, Z.shape[0], batch_size):
            Z_ = Z[i : i + batch_size]
            term_1 = torch.bmm(
                torch.unsqueeze((torch.unsqueeze(beta, dim=0) * torch_rbf_kernel(Z_, Z, gamma)), dim=1),
                torch.unsqueeze(Z_, dim=1) - torch.unsqueeze(Z, dim=0),
            )
            if alpha is not None:
                term_2 = -2 * torch.bmm(
                    torch.unsqueeze(alpha * torch_rbf_kernel(Z_, X, gamma), dim=1),
                    torch.unsqueeze(Z_, dim=1) - torch.unsqueeze(X, dim=0),
                )
            else:
                term_2 = -2 * torch.bmm(
                    torch.unsqueeze(torch_rbf_kernel(Z_, X, gamma) / self.num_points, dim=1),
                    torch.unsqueeze(Z_, dim=1) - torch.unsqueeze(X, dim=0),
                )
            grad_Z[i : i + batch_size] = (
                -2 * gamma * torch.unsqueeze(beta[i : i + batch_size], dim=1) * torch.squeeze(term_1 + term_2)
            )

        Z = Z - step_size * grad_Z
        self.z = Z

    def _update_z(self, alpha: float, X: Any, step_size: float):
        """Fix beta and update Z using gradient descent.

        Parameters
        ----------
        alpha : int
            Normalization factor.
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        step_size : float
            Step size for gradient descent.
        """
        gamma = self.gamma
        Z = self.z
        beta = self.beta

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self._device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).double()

        grad_Z = torch.zeros_like(Z)

        for i in range(Z.shape[0]):
            z_i = Z[i, :].reshape(1, -1)
            term_1 = (beta * torch_rbf_kernel(z_i, Z, gamma)) @ (z_i - Z)
            if alpha is not None:
                term_2 = -2 * (alpha * torch_rbf_kernel(z_i, X, gamma)) @ (z_i - X)
            else:
                term_2 = -2 * (torch_rbf_kernel(z_i, X, gamma) / self.num_points) @ (z_i - X)
            grad_Z[i, :] = -2 * gamma * beta[i] * (term_1 + term_2)

        Z = Z - step_size * grad_Z
        self.z = Z

    def _inner_prod_with_X(self, X: Any) -> float:
        """Compute the inner product between RKME specification and X

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.

        Returns
        -------
        float
            The inner product between RKME specification and X
        """
        beta = self.beta.reshape(1, -1).double().to(self._device)
        Z = self.z.double().to(self._device)
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).double()

        v = torch_rbf_kernel(Z, X, self.gamma) * beta.T
        v = torch.sum(v, axis=0)
        return v.detach().cpu().numpy()

    def _sampling_candidates(self, N: int) -> np.ndarray:
        """Generate a large set of candidates as preparation for herding

        Parameters
        ----------
        N : int
            The number of herding candidates.

        Returns
        -------
        np.ndarray
            The herding candidates.
        """
        beta = self.beta
        beta[beta < 0] = 0  # currently we cannot use negative weight
        beta = beta / torch.sum(beta)
        sample_assign = torch.multinomial(beta, N, replacement=True)

        sample_list = []
        for i, n in Counter(np.array(sample_assign.cpu())).items():
            for _ in range(n):
                sample_list.append(
                    torch.normal(mean=self.z[i].reshape(self.z[i].shape[0], -1), std=0.25).reshape(1, -1)
                )
        if len(sample_list) > 1:
            return torch.cat(sample_list, axis=0)
        elif len(sample_list) == 1:
            return sample_list[0]
        else:
            logger.warning("Not enough candidates for herding!")

    def inner_prod(self, Phi2: RKMETableSpecification) -> float:
        """Compute the inner product between two RKME specifications

        Parameters
        ----------
        Phi2 : RKMETableSpecification
            The other RKME specification.

        Returns
        -------
        float
            The inner product between two RKME specifications.
        """
        beta_1 = self.beta.reshape(1, -1).double().to(self._device)
        beta_2 = Phi2.beta.reshape(1, -1).double().to(self._device)
        Z1 = self.z.double().reshape(self.z.shape[0], -1).to(self._device)
        Z2 = Phi2.z.double().reshape(Phi2.z.shape[0], -1).to(self._device)
        v = torch.sum(torch_rbf_kernel(Z1, Z2, self.gamma) * (beta_1.T @ beta_2))

        return float(v)

    def dist(self, Phi2: RKMETableSpecification, omit_term1: bool = False) -> float:
        """Compute the Maximum-Mean-Discrepancy(MMD) between two RKME specifications

        Parameters
        ----------
        Phi2 : RKMETableSpecification
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

        return float(term1 - 2 * term2 + term3)

    def herding(self, T: int) -> np.ndarray:
        """Iteratively sample examples from an unknown distribution with the help of its RKME specification

        Parameters
        ----------
        T : int
            Total iteration number for sampling.

        Returns
        -------
        np.ndarray
            A collection of examples which approximate the unknown distribution.
        """
        # Flatten z
        Z_shape = self.z.shape
        self.z = self.z.reshape(self.z.shape[0], -1)

        Nstart = 100 * T
        Xstart = self._sampling_candidates(Nstart).to(self._device)
        D = self.z[0].shape[0]
        S = torch.zeros((T, D)).to(self._device)
        fsX = torch.from_numpy(self._inner_prod_with_X(Xstart)).to(self._device)
        fsS = torch.zeros(Nstart).to(self._device)
        for i in range(T):
            if i > 0:
                fsS = torch.sum(torch_rbf_kernel(S[:i, :], Xstart, self.gamma), axis=0)
            fs = (i + 1) * fsX - fsS
            idx = torch.argmax(fs)
            S[i, :] = Xstart[idx, :]

        # Reshape to orignial dimensions
        self.z = self.z.reshape(Z_shape)
        S_shape = tuple([S.shape[0]] + list(Z_shape)[1:])
        S = S.reshape(S_shape)

        return S

    def save(self, filepath: str):
        """Save the computed RKME specification to a specified path in JSON format.

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
        """Load a RKME specification file in JSON format from the specified path.

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
            rkme_load["z"] = torch.from_numpy(np.array(rkme_load["z"]))
            rkme_load["beta"] = torch.from_numpy(np.array(rkme_load["beta"]))

            for d in self.get_states():
                if d in rkme_load.keys():
                    setattr(self, d, rkme_load[d])
            return True
        else:
            return False


class RKMEStatSpecification(RKMETableSpecification):
    """nickname for RKMETableSpecification, for compatibility currently.
    TODO: modify all learnware in database and remove this nickname
    """

    def __init__(self, gamma: float = 0.1, cuda_idx: int = -1):
        super(RKMEStatSpecification, self).__init__(gamma=gamma, cuda_idx=cuda_idx)
        super(RKMETableSpecification, self).__init__(type=RKMETableSpecification.__name__)


def torch_rbf_kernel(x1, x2, gamma) -> torch.Tensor:
    """Use pytorch to compute rbf_kernel function at faster speed.

    Parameters
    ----------
    x1 : torch.Tensor
            First vector in the rbf_kernel
    x2 : torch.Tensor
            Second vector in the rbf_kernel
    gamma : float
            Bandwidth in gaussian kernel

    Returns
    -------
    torch.Tensor
            The computed rbf_kernel value at x1, x2.
    """
    x1 = x1.double()
    x2 = x2.double()
    X12norm = torch.sum(x1**2, 1, keepdim=True) - 2 * x1 @ x2.T + torch.sum(x2**2, 1, keepdim=True).T
    return torch.exp(-X12norm * gamma)


def rkme_solve_qp(K: np.ndarray, C: np.ndarray):
    """Solver for the following quadratic programming(QP) problem:
        - min   1/2 x^T K x - C^T x
        s.t     1^T x - 1 = 0
                - I x <= 0

    Parameters
    ----------
    K : np.ndarray
            Parameter in the quadratic term.
    C : np.ndarray
            Parameter in the linear term.

    Returns
    -------
    torch.tensor
            Solution to the QP problem.
    """
    if torch.is_tensor(K):
        K = K.cpu().numpy()
    if torch.is_tensor(C):
        C = C.cpu().numpy()
    C = C.reshape(-1)

    n = K.shape[0]
    P = np.array(K)
    P = scipy.sparse.csc_matrix(P)
    q = np.array(-C)
    G = np.array(-np.eye(n))
    G = scipy.sparse.csc_matrix(G)
    h = np.array(np.zeros((n, 1)))
    A = np.array(np.ones((1, n)))
    A = scipy.sparse.csc_matrix(A)
    b = np.array(np.ones((1, 1)))
    problem = Problem(P, q, G, h, A, b)
    solution = solve_problem(problem, solver="clarabel")
    w = solution.x
    w = torch.from_numpy(w).reshape(-1)
    return w, solution.obj
