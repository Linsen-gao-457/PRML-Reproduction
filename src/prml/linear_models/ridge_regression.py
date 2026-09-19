import numpy as np


class RidgeRegression:
    def __init__(self, regularization):
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self.regularization = regularization
        self._coef = None

    def fit(self, targets, design_matrix):
        design_matrix = np.asarray(design_matrix)
        targets = np.asarray(targets)
        identity = np.eye(design_matrix.shape[1])
        lhs = design_matrix.T @ design_matrix + self.regularization * identity
        rhs = design_matrix.T @ targets
        self._coef = np.linalg.solve(lhs, rhs)
        return self

    def predict(self, design_matrix):
        if self._coef is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(design_matrix) @ self._coef
