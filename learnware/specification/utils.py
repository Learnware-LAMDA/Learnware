import numpy as np
import torch
import random
from cvxopt import solvers, matrix
from .base import BaseStatSpecification


def setup_seed(seed):
    """
		Fix a random seed for addressing reproducibility issues.
	
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
    """
		Let users choose compuational device between CPU or GPU.

	Parameters
	----------
	cuda_idx : int, optional
		GPU index, by default -1 which stands for using CPU instead.

	Returns
	-------
	torch.device
		A torch.device object
	"""
    if cuda_idx != -1:
        device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = torch.device("cpu")
    return device


def torch_rbf_kernel(x1, x2, gamma) -> torch.Tensor:
    """
		Use pytorch to compute rbf_kernel function at faster speed.

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
    X12norm = torch.sum(x1 ** 2, 1, keepdim=True) - 2 * x1 @ x2.T + torch.sum(x2 ** 2, 1, keepdim=True).T
    return torch.exp(-X12norm * gamma)


def solve_qp(K: np.ndarray, C: np.ndarray):
    """
		Solver for the following quadratic programming(QP) problem:
		- min    1/2 x^T K x - C^T x
    	  s.t    1^T x - 1 = 0
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
    n = K.shape[0]
    P = matrix(K.cpu().numpy())
    q = matrix(-C.cpu().numpy())
    G = matrix(-np.eye(n))
    h = matrix(np.zeros((n, 1)))
    A = matrix(np.ones((1, n)))
    b = matrix(np.ones((1, 1)))

    solvers.options["show_progress"] = False
    sol = solvers.qp(P, q, G, h, A, b)  # Requires the sum of x to be 1
    # sol = solvers.qp(P, q, G, h) # Otherwise
    w = np.array(sol["x"])
    w = torch.from_numpy(w).reshape(-1)

    return w


def generate_stat_spec(X: np.ndarray) -> BaseStatSpecification:
    """
		Interface for users to generate statistical specification.
		Return a StatSpecification object, use .save() method to save as npy file.


	Parameters
	----------
	X : np.ndarray
		Raw data in np.ndarray format.
		Size of array: (n*d)

	Returns
	-------
	StatSpecification
		A StatSpecification object
	"""
    return None
