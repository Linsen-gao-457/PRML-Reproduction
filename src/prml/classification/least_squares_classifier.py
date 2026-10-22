import numpy as np


class LeastSquaresClassifier:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._intercept = None
        self._weights = None

    def fit(self, features, targets):
        pass

    def predict(self, feature):
        pass
