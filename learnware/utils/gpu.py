import random
import numpy as np
from .import_utils import is_torch_available


def setup_seed(seed):
    """Fix a random seed for addressing reproducibility issues.

    Parameters
    ----------
    seed : int
            Random seed for torch, torch.cuda, numpy, random and cudnn libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    if is_torch_available(verbose=False):
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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


def allocate_cuda_idx():
    if is_torch_available(verbose=False):
        import torch

        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        cuda_count = 0

    if cuda_count == 0:
        return -1
    return np.random.randint(0, cuda_count)
