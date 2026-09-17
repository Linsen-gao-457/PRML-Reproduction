import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = None

    def fit(self, design_matrix, targets):
        self.coef_, _, _, _ = np.linalg.lstsq(design_matrix, targets)
        return self

    def predict(self, design_matrix):
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        return design_matrix @ self.coef_
