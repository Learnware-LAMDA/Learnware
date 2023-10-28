import numpy as np
from .data import ProductIterator, DiagIterator, print_timings

__all__ = ('create_h5py_dataset', 'save_K')


def create_h5py_dataset(f, batch_size, name, diag, N, N2):
    """
    Creates a dataset named `name` on `f`, with chunks of size `batch_size`.
    The chunks have leading dimension 1, so as to accommodate future resizing
    of the leading dimension of the dataset (which starts at 1).
    """
    if diag:
        chunk_shape = (1, min(batch_size, N))
        shape = (1, N)
        maxshape = (None, N)
    else:
        chunk_shape = (1, min(batch_size, N), min(batch_size, N2))
        shape = (1, N, N2)
        maxshape = (None, N, N2)
    return f.create_dataset(name, shape=shape, dtype=np.float32,
                            fillvalue=np.nan, chunks=chunk_shape,
                            maxshape=maxshape)


def save_K(f, kern, name, X, X2, diag, batch_size, worker_rank=0, n_workers=1,
           print_interval=2.):
    """
    Saves a kernel to the h5py file `f`. Creates its dataset with name `name`
    if necessary.
    """
    if name in f.keys():
        print("Skipping {} (group exists)".format(name))
        return
    else:
        N = len(X)
        N2 = N if X2 is None else len(X2)
        out = create_h5py_dataset(f, batch_size, name, diag, N, N2)

    if diag:
        # Don't split the load for diagonals, they are cheap
        it = DiagIterator(batch_size, X, X2)
    else:
        it = ProductIterator(batch_size, X, X2, worker_rank=worker_rank,
                             n_workers=n_workers)
    it = print_timings(it, desc=f"{name} (worker {worker_rank}/{n_workers})",
                       print_interval=print_interval)

    for same, (i, (x, _y)), (j, (x2, _y2)) in it:
        k = kern(x, x2, same, diag)
        if np.any(np.isinf(k)) or np.any(np.isnan(k)):
            print(f"About to write a nan or inf for {i},{j}")
            import ipdb; ipdb.set_trace()

        if diag:
            out[0, i:i+len(x)] = k
        else:
            out[0, i:i+len(x), j:j+len(x2)] = k
