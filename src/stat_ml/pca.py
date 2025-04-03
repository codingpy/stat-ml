import numpy as np
from numpy import linalg as LA


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        eigvals, eigvecs = LA.eigh(np.cov(X, rowvar=False))

        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        if self.n_components < 1:
            self.n_components = (
                np.searchsorted(np.cumsum(eigvals / eigvals.sum()), self.n_components)
                + 1
            )

        self.explained_variance = eigvals[: self.n_components]
        self.components = eigvecs[:, : self.n_components]

        return self

    def transform(self, X):
        return X @ self.components
