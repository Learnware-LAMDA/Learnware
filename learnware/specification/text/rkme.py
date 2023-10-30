from sentence_transformers import SentenceTransformer
from ..rkme import RKMEStatSpecification
import numpy as np
import os

class TextRKMEStatSpecification(RKMEStatSpecification):
    """Reduced Kernel Mean Embedding (RKME) Specification for Text"""

    def generate_stat_spec_from_text(
        self,
        X: list,
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
        
        # Sentence embedding for Text
        X = sentence_embedding(X)

        return self.generate_stat_spec_from_data(
            X, K, step_size,steps,
            nonnegative_beta,
            reduce,
        )

def sentence_embedding(X):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    X = model.encode(X)
    X = np.array(X)
    # print(X.shape, np.mean(X, axis=1).shape, np.std(X, axis=1).reshape(X.shape[0], 1).shape)
    # X = (X - np.mean(X, axis=1).reshape(X.shape[0], 1)) / np.std(X, axis=1).reshape(X.shape[0], 1)
    return X
