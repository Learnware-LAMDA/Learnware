import random
import numpy as np
from .import_utils import is_torch_available


def setup_seed(seed):
    import torch

    """Fix a random seed for addressing reproducibility issues.

    Parameters
    ----------
    seed : int
            Random seed for torch, torch.cuda, numpy, random and cudnn libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def choose_device(cuda_idx=-1):
    import torch

    """Let users choose compuational device between CPU or GPU.

    Parameters
    ----------
    cuda_idx : int, optional
            GPU index, by default -1 which stands for using CPU instead.

    Returns
    -------
    torch.device
            A torch.device object
    """
    cuda_idx = int(cuda_idx)
    if cuda_idx == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device_count = torch.cuda.device_count()
        if cuda_idx >= 0 and cuda_idx < device_count:
            device = torch.device(f"cuda:{cuda_idx}")
        else:
            device = torch.device("cuda:0")
    return device


class CudaManager:
    def __init__(self):
        if is_torch_available(verbose=False):
            import torch

            self.cuda_avalable = torch.cuda.is_available()
            self.cuda_count = torch.cuda.device_count() if self.cuda_avalable else 0
        else:
            self.cuda_avalable = False
            self.cuda_count = 0

        self.cur_cuda_idx = 0
        self.stat_spec_cuda = {}

    def reset(self):
        self.cur_cuda_idx = 0
        self.stat_spec_cuda = {}

    def allocate_cuda(self):
        if not self.cuda_avalable:
            return -1

        ret_cuda_idx = self.cur_cuda_idx
        self.cur_cuda_idx = (self.cur_cuda_idx + 1) % self.cuda_count
        return ret_cuda_idx

    def allocate_stat_spec_cuda(self, stat_spec):
        if stat_spec.type not in self.stat_spec_cuda:
            self.stat_spec_cuda[stat_spec.type] = self.allocate_cuda()

        return self.stat_spec_cuda[stat_spec.type]


cuda_manager = CudaManager()
