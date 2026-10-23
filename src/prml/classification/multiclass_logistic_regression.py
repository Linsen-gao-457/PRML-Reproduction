"""Multiclass logistic regression using the softmax function."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


class MulticlassLogisticRegression:
    """Multiclass logistic regression fitted by maximum likelihood.

    The final class is used as the reference class, with its weight vector
    fixed at zero. This removes the redundant softmax parameterization.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include a constant bias term.
    tolerance : float, default=1e-8
        Gradient tolerance used by the optimizer.
    max_iterations : int, default=1000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,) or None
        Class labels observed during fitting.
    coef_ : numpy.ndarray of shape (n_classes, n_features) or None
        Feature coefficients for each class.
    intercept_ : numpy.ndarray of shape (n_classes,) or None
        Intercept for each class.
    n_iterations_ : int or None
        Number of optimization iterations.
    """

    def __init__(
        self,
        fit_intercept=True,
        tolerance=1.0e-8,
        max_iterations=1000,
    ):
        """Initialize an unfitted classifier."""
        if not np.isscalar(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive")

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")

        self.fit_intercept = bool(fit_intercept)
        self.tolerance = float(tolerance)
        self.max_iterations = max_iterations

        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iterations_ = None

        self._weights = None
        self._n_features = None

    def fit(self, features, targets):
        """Fit the multiclass softmax model.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Training observations.
        targets : array_like of shape (n_samples,)
            Class labels.

        Returns
        -------
        MulticlassLogisticRegression
            Fitted classifier.
        """
        features, targets = self._validate_training_data(
            features,
            targets,
        )

        self.classes_, encoded_targets = np.unique(
            targets,
            return_inverse=True,
        )

        if self.classes_.size < 2:
            raise ValueError("targets must contain at least two classes")

        design_matrix = self._create_design_matrix(features)
        number_of_free_classes = self.classes_.size - 1
        number_of_parameters = number_of_free_classes * design_matrix.shape[1]

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=np.zeros(number_of_parameters),
            args=(
                design_matrix,
                encoded_targets,
                self.classes_.size,
            ),
            jac=self._gradient,
            method="BFGS",
            options={
                "gtol": self.tolerance,
                "maxiter": self.max_iterations,
            },
        )

        gradient_norm = np.linalg.norm(
            result.jac,
            ord=np.inf,
        )

        if not result.success and gradient_norm > 1.0e-6:
            raise RuntimeError(
                f"multiclass logistic-regression optimization failed: {result.message}"
            )

        free_weights = result.x.reshape(
            number_of_free_classes,
            design_matrix.shape[1],
        )

        # The final class is the reference class.
        self._weights = np.vstack(
            (
                free_weights,
                np.zeros(
                    (1, design_matrix.shape[1]),
                    dtype=float,
                ),
            )
        )

        self._n_features = features.shape[1]
        self.n_iterations_ = result.nit

        if self.fit_intercept:
            self.intercept_ = self._weights[:, 0].copy()
            self.coef_ = self._weights[:, 1:].copy()
        else:
            self.intercept_ = np.zeros(
                self.classes_.size,
                dtype=float,
            )
            self.coef_ = self._weights.copy()

        return self

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

    @staticmethod
    def _unpack_free_weights(
        flat_weights,
        number_of_classes,
        number_of_columns,
    ):
        free_weights = flat_weights.reshape(
            number_of_classes - 1,
            number_of_columns,
        )

        return np.vstack(
            (
                free_weights,
                np.zeros(
                    (1, number_of_columns),
                    dtype=float,
                ),
            )
        )

    @classmethod
    def _negative_log_likelihood(
        cls,
        flat_weights,
        design_matrix,
        targets,
        number_of_classes,
    ):
        weights = cls._unpack_free_weights(
            flat_weights,
            number_of_classes,
            design_matrix.shape[1],
        )
        scores = design_matrix @ weights.T

        return np.sum(
            logsumexp(scores, axis=1) - scores[np.arange(targets.size), targets]
        )

    @classmethod
    def _gradient(
        cls,
        flat_weights,
        design_matrix,
        targets,
        number_of_classes,
    ):
        weights = cls._unpack_free_weights(
            flat_weights,
            number_of_classes,
            design_matrix.shape[1],
        )

        scores = design_matrix @ weights.T
        probabilities = softmax(scores, axis=1)

        target_matrix = np.eye(
            number_of_classes,
            dtype=float,
        )[targets]

        gradient = (probabilities - target_matrix).T @ design_matrix

        # The last class is fixed, so only return free gradients.
        return gradient[:-1].ravel()

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
