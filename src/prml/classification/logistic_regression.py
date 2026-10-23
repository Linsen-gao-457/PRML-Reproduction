"""Binary logistic regression."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticRegression:
    """Binary logistic regression fitted by maximum likelihood.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include a constant bias term.
    tolerance : float, default=1e-8
        Gradient tolerance passed to the optimizer.
    max_iterations : int, default=1000
        Maximum number of optimizer iterations.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (2,) or None
        Class labels observed during fitting.
    coef_ : numpy.ndarray of shape (n_features,) or None
        Fitted feature coefficients.
    intercept_ : float or None
        Fitted intercept.
    n_iterations_ : int or None
        Number of optimizer iterations.
    """

    def __init__(
        self,
        fit_intercept=True,
        tolerance=1.0e-8,
        max_iterations=1000,
    ):
        """Initialize an unfitted binary logistic-regression model."""
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
        """Fit the model by minimizing binary cross-entropy.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Training observations.
        targets : array_like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        LogisticRegression
            Fitted classifier.

        Raises
        ------
        ValueError
            If the inputs are invalid or do not contain exactly two classes.
        RuntimeError
            If numerical optimization fails to converge.
        """
        features, targets = self._validate_training_data(
            features,
            targets,
        )

        self.classes_, encoded_targets = np.unique(
            targets,
            return_inverse=True,
        )

        if self.classes_.size != 2:
            raise ValueError("binary logistic regression requires exactly two classes")

        encoded_targets = encoded_targets.astype(float)
        design_matrix = self._create_design_matrix(features)

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=np.zeros(design_matrix.shape[1]),
            args=(design_matrix, encoded_targets),
            jac=self._gradient,
            method="BFGS",
            options={
                "gtol": self.tolerance,
                "maxiter": self.max_iterations,
            },
        )

        gradient_norm = np.linalg.norm(
            self._gradient(
                result.x,
                design_matrix,
                encoded_targets,
            ),
            ord=np.inf,
        )

        if not result.success and gradient_norm > 1.0e-6:
            raise RuntimeError(
                f"logistic-regression optimization failed: {result.message}"
            )

        self._weights = result.x
        self._n_features = features.shape[1]
        self.n_iterations_ = result.nit

        if self.fit_intercept:
            self.intercept_ = float(self._weights[0])
            self.coef_ = self._weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights.copy()

        return self

    def decision_function(self, features):
        """Return the linear score before applying the sigmoid.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Observations to evaluate.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Linear scores.
        """
        features = self._validate_prediction_data(features)
        design_matrix = self._create_design_matrix(features)
        return design_matrix @ self._weights

    def predict_proba(self, features):
        """Return probabilities for both classes.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Observations to evaluate.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Columns correspond to ``classes_[0]`` and ``classes_[1]``.
        """
        positive_probability = expit(self.decision_function(features))

        return np.column_stack(
            (
                1.0 - positive_probability,
                positive_probability,
            )
        )

    def predict(self, features):
        """Predict the class label of each observation.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Observations to classify.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        positive_class = self.decision_function(features) >= 0.0
        return self.classes_[positive_class.astype(int)]

    @staticmethod
    def _negative_log_likelihood(
        weights,
        design_matrix,
        targets,
    ):
        scores = design_matrix @ weights

        # logaddexp gives a stable implementation of log(1 + exp(score)).
        return np.sum(np.logaddexp(0.0, scores) - targets * scores)

    @staticmethod
    def _gradient(
        weights,
        design_matrix,
        targets,
    ):
        probabilities = expit(design_matrix @ weights)
        return design_matrix.T @ (probabilities - targets)

    def _create_design_matrix(self, features):
        if not self.fit_intercept:
            return features

        return np.column_stack(
            (
                np.ones(features.shape[0]),
                features,
            )
        )

    def _validate_training_data(self, features, targets):
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
