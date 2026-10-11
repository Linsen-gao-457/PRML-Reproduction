import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None

    def fit(self, design_matrix, targets):
        if design_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")

        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")

        if design_matrix.shape[0] != targets.size:
            raise ValueError(
                "design_matrix and targets must have the same number of observations"
            )
        self.coef_, _, _, _ = np.linalg.lstsq(design_matrix, targets)
        return self

    def predict(self, design_matrix):
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        if design_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")
        return np.dot(design_matrix, self.coef_)
