from __future__ import annotations

import os
import copy
import json
import torch
import codecs
import numpy as np

from .base import SystemStatSpecification
from ..regular import RKMETableSpecification
from ..regular.table.rkme import torch_rbf_kernel
from ...utils import choose_device, allocate_cuda_idx


class HeteroMapTableSpecification(SystemStatSpecification):
    """Heterogeneous Map-Table Specification"""

    def __init__(self, gamma: float = 0.1, cuda_idx: int = None):
        """Initializing HeteroMapTableSpecification parameters.

        Parameters
        ----------
        gamma : float
            Bandwidth in gaussian kernel, by default 0.1.
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used.
        """
        self.z = None
        self.beta = None
        self.embedding = None
        self.weight = None
        self.gamma = gamma
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        torch.cuda.empty_cache()
        self._device = choose_device(cuda_idx=self._cuda_idx)
        super(HeteroMapTableSpecification, self).__init__(type=self.__class__.__name__)

    @property
    def device(self):
        return self._device

    def get_z(self) -> np.ndarray:
        """Move z(RKME reduced set points) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of z in CPU memory.
        """
        return self.z.detach().cpu().numpy()

    def get_beta(self) -> np.ndarray:
        """Move beta(RKME weights weights) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of beta in CPU memory.
        """
        return self.beta.detach().cpu().numpy()

    def generate_stat_spec_from_system(self, heter_embedding: np.ndarray, rkme_spec: RKMETableSpecification):
        """Construct heterogeneous map-table specification from RKME specification and embedding genereated by heterogeneous market mapping.

        Parameters
        ----------
        heter_embedding : np.ndarray
            Embedding genereated by the heterogeneous market mapping.
        rkme_spec : RKMETableSpecification
            The RKME specification.
        """
        self.beta = rkme_spec.beta.to(self._device)
        self.z = torch.from_numpy(heter_embedding).double().to(self._device)

    def inner_prod(self, Embed2: HeteroMapTableSpecification) -> float:
        """Compute the inner product between two HeteroMapTableSpecifications

        Parameters
        ----------
        Embed2 : HeteroMapTableSpecification
            The other HeteroMapTableSpecification.

        Returns
        -------
        float
            The inner product between two HeteroMapTableSpecifications.
        """
        beta_1 = self.beta.reshape(1, -1).double().to(self._device)
        beta_2 = Embed2.beta.reshape(1, -1).double().to(self._device)
        Z1 = self.z.double().reshape(self.z.shape[0], -1).to(self._device)
        Z2 = Embed2.z.double().reshape(Embed2.z.shape[0], -1).to(self._device)
        v = torch.sum(torch_rbf_kernel(Z1, Z2, self.gamma) * (beta_1.T @ beta_2))

        return float(v)

    def dist(self, Embed2: HeteroMapTableSpecification, omit_term1: bool = False) -> float:
        """Compute the Maximum-Mean-Discrepancy(MMD) between two HeteroMapTableSpecifications

        Parameters
        ----------
        Phi2 : HeteroMapTableSpecification
            The other HeteroMapTableSpecification.
        omit_term1 : bool, optional
            True if the inner product of self with itself can be omitted, by default False.
        """
        term1 = 0 if omit_term1 else self.inner_prod(self)
        term2 = self.inner_prod(Embed2)
        term3 = Embed2.inner_prod(Embed2)

        return float(term1 - 2 * term2 + term3)

    def load(self, filepath: str) -> bool:
        """Load a HeteroMapTableSpecification file in JSON format from the specified path.

        Parameters
        ----------
        filepath : str
            The specified loading path.

        Returns
        -------
        bool
            True if the HeteroMapTableSpecification is loaded successfully.
        """
        load_path = filepath
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            embedding_load = json.loads(obj_text)
            embedding_load["z"] = torch.from_numpy(np.array(embedding_load["z"]))
            embedding_load["beta"] = torch.from_numpy(np.array(embedding_load["beta"]))

            for d in self.get_states():
                if d in embedding_load.keys():
                    setattr(self, d, embedding_load[d])

            return True
        else:
            return False

    def save(self, filepath: str) -> bool:
        """Save the computed HeteroMapTableSpecification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path.
        """
        save_path = filepath
        embedding_to_save = self.get_states()
        if torch.is_tensor(embedding_to_save["z"]):
            embedding_to_save["z"] = embedding_to_save["z"].detach().cpu().numpy()
        embedding_to_save["z"] = embedding_to_save["z"].tolist()
        if torch.is_tensor(embedding_to_save["beta"]):
            embedding_to_save["beta"] = embedding_to_save["beta"].detach().cpu().numpy()
        embedding_to_save["beta"] = embedding_to_save["beta"].tolist()
        with codecs.open(save_path, "w", encoding="utf-8") as fout:
            json.dump(embedding_to_save, fout, separators=(",", ":"))
