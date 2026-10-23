import numpy as np


class LeastSquaresClassifier:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.intercept_ = None
        self.coef_ = None
        self.classes_ = None
        self._weights = None
        self.n_features = None

    def fit(self, features, targets):
        features, targets = self._validate_training_data(features, targets)
        self.classes_, indices = np.unique(targets, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("targets must contain at least two classes")
        targets_matrix = np.eye(N=self.classes_.size, dtype=float)[indices]
        design_matrix = self._create_design_matrix(features)
        self._weights, _, _, _ = np.linalg.lstsq(
            design_matrix, targets_matrix, rcond=None
        )
        self.n_features = features.shape[1]
        if self.fit_intercept:
            self.intercept_ = self._weights[0].copy()
            self._weights = self._weights[1:].T.copy()
        else:
            self.intercept_ = np.zeros(
                self.classes_.size,
                dtype=float,
            )
            self.coef_ = self._weights.T.copy()
        return self

    def predict(self, feature):
        pass

    def _create_design_matrix(self, features):
        if not self.fit_intercept:
            return features
        else:
            return np.column_stack(
                (
                    np.ones(features.shape[0]),
                    features,
                )
            )

    def _validate_training_data(features, targets):
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets)

        if features.ndim != 2:
            raise ValueError("features must be two-dimensional")

        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")

        if features.shape[0] != targets.size:
            raise ValueError(
                "features and targets must have the same number of samples"
            )

        if features.shape[0] == 0:
            raise ValueError("training data must not be empty")

        if features.shape[1] == 0:
            raise ValueError("features must contain at least one feature")

        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")

        return features, targets
