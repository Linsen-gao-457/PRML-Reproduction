"""Multiclass logistic regression using the softmax function."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


class MulticlassLogisticRegression:
    """Multiclass logistic regression fitted using Newton/IRLS updates.

    Fix the final class's weight vector to zero to remove the redundant
    softmax parameterization. Use a coupled block Hessian and backtracking
    line search to minimize the negative log-likelihood.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept.
    tolerance : float, default=1e-8
        Convergence threshold for the infinity norm of the gradient
        of the mean negative log-likelihood.
    max_iterations : int, default=100
        Maximum number of Newton/IRLS updates.
    damping : float, default=1e-8
        Positive value added to the Hessian diagonal when computing
        updates. This stabilizes the solve without adding a penalty
        to the loss.

    Attributes
    ----------
    classes_ : ndarray of shape (k_classes,)
        Class labels in probability-column order.
    k_classes_ : int
        Number of training classes.
    m_features_ : int
        Number of input features.
    coef_ : ndarray of shape (k_classes, m_features)
        Learned coefficients. The final row is zero.
    intercept_ : ndarray of shape (k_classes,)
        Learned intercepts. The final entry is zeitro.
    n_iterations_ : int
        Number of accepted parameter updates.
    converged_ : bool
        Whether the gradient convergence criterion was satisfied.
    loss_history_ : list of float
        Mean negative log-likelihood at initialization and after each
        accepted update.

    Notes
    -----
    The dense Hessian requires memory proportional to
    ``((k_classes - 1) * (m_features + fit_intercept)) ** 2``.
    This implementation is intended for small educational examples.

    Unregularized logistic regression may not have a finite optimum
    on separable data. Numerical damping does not resolve that issue.
    """

    def __init__(
        self,
        fit_intercept=True,
        tolerance=1.0e-8,
        max_iterations=1000,
        damping=0,
    ):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")

        self.fit_intercept = bool(fit_intercept)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.damping = damping

        self.coef_ = None
        self.intercept_ = None

    def fit(self, features, targets):
        pass

    def decision_function(self, features):
        """Return the linear score for each class."""
        features = self._validate_prediction_data(features)
        design_matrix = self._create_design_matrix(features)
        return np.dot(design_matrix, self._weights.T)

    def predict_proba(self, features):
        """Return the posterior class probabilities."""
        scores = self.decision_function(features)
        return softmax(scores, axis=1)

    def predict(self, features):
        """Predict the class label for every observation."""
        probabilities = self.predict_proba(features)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def _create_design_matrix(self, features):
        if not self.fit_intercept:
            return features

        return np.column_stack(
            (
                np.ones(features.shape[0]),
                features,
            )
        )

    @staticmethod
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

    def _validate_prediction_data(self, features):
        if self._weights is None:
            raise RuntimeError("fit must be called before prediction")

        features = np.asarray(features, dtype=float)

        if features.ndim != 2:
            raise ValueError("features must be two-dimensional")

        if features.shape[1] != self._n_features:
            raise ValueError(f"features must contain {self._n_features} columns")

        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")

        return features
