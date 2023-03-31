import os
import copy
import torch
import faiss
import json
import codecs
import numpy as np
from .base import BaseStatSpecification
from .utils import setup_seed, choose_device, torch_rbf_kernel, solve_qp
from typing import Tuple, Any, List, Union, Dict
from learnware.config import C


class RKMESpecification(BaseStatSpecification):
    """Reduced-set Kernel Mean Embedding(RKME) Specification
        
    """
    def __init__(self, gamma: float = 0.1, cuda_idx: int = -1):
        """Initializing RKME parameters.

        Parameters
        ----------
        gamma : float
            Bandwidth in gaussian kernel, by default 0.1.
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used.
        """
        self.Z = []
        self.beta = []
        self.gamma = gamma
        self.num_points = 0
        self.cuda_idx = cuda_idx
        torch.cuda.empty_cache()
        self.device = choose_device(cuda_idx=cuda_idx)
        setup_seed(0)
    
    def get_beta(self) -> torch.tensor:
        """Move beta(RKME weights) back to memory accessible to the CPU.

        Returns
        -------
        torch.tensor
            A copy of beta in CPU memory.
        """
        return self.beta.detach().cpu()
    
    def get_z(self) -> torch.tensor:
        """Move z(RKME reduced set points) back to memory accessible to the CPU.

        Returns
        -------
        torch.tensor
            A copy of z in CPU memory.
        """
        return self.z.detach().cpu()
    
    def generate_stat_spec_from_data(self, X: np.ndarray, K: int, step_size: float, steps: int, reduce: bool = True, nonnegative_beta: bool = False):
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
        reduce : bool, optional
            Whether shrink original data to a smaller set, by default True
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        """
        alpha = None
        self.num_points = X.shape[0]

        if not reduce:
            self.z = X
            self.beta = 1 / self.num_points * np.ones(self.num_points)
            self.z = torch.from_numpy(self.z).double().to(self.device)
            self.beta = torch.from_numpy(self.beta).double().to(self.device)
            return

        # Initialize Z by clustering, utiliing faiss to speed up the process.
        self._init_z_by_faiss(X, K)
        self._update_beta(X, nonnegative_beta)

        # Alternating optimize Z and beta
        for i in range(steps):
            self._update_z(alpha, X, step_size)
            self._update_beta(X, nonnegative_beta)
    
    def _init_z_by_faiss(self, X: Any, K: int):
        """Intialize Z by faiss clustering.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        K : int
            Size of the construced reduced set.
        """
        numDim = X.shape[1]
        kmeans = faiss.Kmeans(numDim, K, niter=100, verbose=False)
        kmeans.train(X)
        center = torch.from_numpy(kmeans.centroids).double()
        self.z = center
    
    def _update_beta(self, X: Any, nonnegative_beta: bool = False):
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
        Z = Z.to(self.device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device).double()
        K = torch_rbf_kernel(Z, Z, gamma=self.gamma).to(self.device)
        C = torch_rbf_kernel(Z, X, gamma=self.gamma).to(self.device)
        C = torch.sum(C, dim=1) / X.shape[0]

        if nonnegative_beta:
            beta = solve_qp(K, C).to(self.device)
        else:
            beta = torch.linalg.inv(K + torch.eye(K.shape[0]).to(self.device) * 1e-5) @ C

        self.beta = beta
    
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
        Z = Z.to(self.device).double()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self.device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device).double()

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

    def eval_Phi(self, Phi2: RKMESpecification) -> float:
        """Compute the inner product between two RKME specifications
        
        Parameters
        ----------
        Phi2 : RKMESpecification
            The other RKME specification.
        
        Returns
        -------
        float
            The inner product between two RKME specifications.
        """
        beta_1 = self.beta.reshape(1, -1).double().to(self.device)
        beta_2 = Phi2.beta.reshape(1, -1).double().to(self.device)
        Z1 = self.z.double().to(self.device)
        Z2 = Phi2.z.double().to(self.device)

        v = torch.sum(torch_rbf_kernel(Z1, Z2, self.gamma) * (beta_1.T @ beta_2))
        return float(v)
    
    def MMD(self, Phi2: RKMESpecification, omit_term1: bool = False) -> float:
        """Compute the Maximum-Mean-Discrepancy(MMD) between two RKME specifications

        Parameters
        ----------
        Phi2 : RKMESpecification
            The other RKME specification.
        omit_term1 : bool, optional
            True if the inner product of self with itself can be omitted, by default False
        """
        if omit_term1:
            term1 = 0
        else:
            term1 = self.eval_Phi(self)
        term2 = self.eval_Phi(Phi2)
        term3 = Phi2.eval_Phi(Phi2)

        return float(term1 - 2 * term2 + term3)
    
    def generate_stat_spec_from_data(self, X: np.ndarray):
        return super().generate_stat_spec_from_data(X)
    
    def save(self, filepath: str):
        """Save the computed RKME specification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path.
        """
        save_path = os.path.join(C.specification_path, f"{filepath}.json")
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
        load_path = os.path.join(C.specification_path, f"{filepath}.json")
        if os.path.exists(load_path):
            obj_text = codecs.open(load_path, "r", encoding="utf-8").read()
            rkme_load = json.loads(obj_text)
            rkme_load["device"] = choose_device(rkme_load["cuda_idx"])
            rkme_load["z"] = torch.from_numpy(np.array(rkme_load["z"]))
            rkme_load["beta"] = torch.from_numpy(np.array(rkme_load["beta"]))

            for d in self.__dir__():
                if d in rkme_load.keys():
                    setattr(self, d, rkme_load[d])
            return True
        else:
            return False