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
from ...utils import choose_device, setup_seed


class HeteroMapTableSpecification(SystemStatSpecification):
    """Heterogeneous Embedding Specification"""

    def __init__(self, gamma: float = 0.1, cuda_idx: int = -1):
        self.z = None
        self.beta = None
        self.embedding = None
        self.weight = None
        self.gamma = gamma
        self.cuda_idx = cuda_idx
        torch.cuda.empty_cache()
        self.device = choose_device(cuda_idx=cuda_idx)
        setup_seed(0)
        super(HeteroMapTableSpecification, self).__init__(type=self.__class__.__name__)

    def get_z(self) -> np.ndarray:
        return self.z.detach().cpu().numpy()

    def get_beta(self) -> np.ndarray:
        return self.beta.detach().cpu().numpy()

    def generate_stat_spec_from_system(self, heter_embedding: np.ndarray, rkme_spec: RKMETableSpecification):
        self.beta = rkme_spec.beta.to(self.device)
        self.z = torch.from_numpy(heter_embedding).double().to(self.device)

    def inner_prod(self, Embed2: HeteroMapTableSpecification) -> float:
        beta_1 = self.beta.reshape(1, -1).double().to(self.device)
        beta_2 = Embed2.beta.reshape(1, -1).double().to(self.device)
        Z1 = self.z.double().reshape(self.z.shape[0], -1).to(self.device)
        Z2 = Embed2.z.double().reshape(Embed2.z.shape[0], -1).to(self.device)
        v = torch.sum(torch_rbf_kernel(Z1, Z2, self.gamma) * (beta_1.T @ beta_2))

        return float(v)

    def dist(self, Embed2: HeteroMapTableSpecification, omit_term1: bool = False) -> float:
        term1 = 0 if omit_term1 else self.inner_prod(self)
        term2 = self.inner_prod(Embed2)
        term3 = Embed2.inner_prod(Embed2)

        return float(term1 - 2 * term2 + term3)

    def load(self, filepath: str) -> bool:
        load_path = filepath
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            embedding_load = json.loads(obj_text)
            embedding_load["device"] = choose_device(embedding_load["cuda_idx"])
            embedding_load["z"] = torch.from_numpy(np.array(embedding_load["z"]))
            embedding_load["beta"] = torch.from_numpy(np.array(embedding_load["beta"]))

            for d in self.__dir__():
                if d in embedding_load.keys():
                    setattr(self, d, embedding_load[d])
            return True
        else:
            return False

    def save(self, filepath: str) -> bool:
        save_path = filepath
        embedding_to_save = copy.deepcopy(self.__dict__)
        if torch.is_tensor(embedding_to_save["z"]):
            embedding_to_save["z"] = embedding_to_save["z"].detach().cpu().numpy()
        embedding_to_save["z"] = embedding_to_save["z"].tolist()
        if torch.is_tensor(embedding_to_save["beta"]):
            embedding_to_save["beta"] = embedding_to_save["beta"].detach().cpu().numpy()
        embedding_to_save["beta"] = embedding_to_save["beta"].tolist()
        embedding_to_save["device"] = "gpu" if embedding_to_save["cuda_idx"] != -1 else "cpu"
        json.dump(
            embedding_to_save,
            codecs.open(save_path, "w", encoding="utf-8"),
            separators=(",", ":"),
        )
