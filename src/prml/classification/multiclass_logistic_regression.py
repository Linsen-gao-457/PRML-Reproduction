"""Multiclass logistic regression using the softmax function."""

import numpy as np
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
        """Fit the model using coupled multiclass IRLS updates.

        Parameters
        ----------
        features : array of shape (n_samples, m_features)
            Finite training features.
        targets : array-like of shape (n_samples,)
            Training labels containing at least two distinct classes.

        Returns
        -------
        self : MulticlassLogisticRegressionIRLS
            Fitted classifier.
        """
        X = np.asarray(features, dtype=float)
        y = np.asarray(targets)

        if X.ndim != 2 or 0 in X.shape:
            raise ValueError("features must be a nonempty 2D array.")
        if y.shape != (X.shape[0],):
            raise ValueError("targets must have shape (n_samples,).")
        if not np.isfinite(X).all():
            raise ValueError("features must contain only finite values.")
        classes, index = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("At least two classes are required.")

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = classes
        self.k_classes = len(classes)
        self.m_features = X.shape[1]
        Z = self._create_design_matrix(X)
        n_samples, m_columns = Z.shape
        number_of_weights_row = self.k_classes - 1
        weights = np.zeros((number_of_weights_row, m_columns))
        targets_matrix = np.eye(self.k_classes)[index]
        free_targets = targets_matrix[:, :number_of_weights_row]

        self.n_iterations_ = 0
        self.converged_ = False
        self.loss_history_ = [self._loss(Z, index, weights)]
        for _ in range(self.max_iterations):
            probabilities = softmax(self._scores(Z, weights), axis=1)
            gradient = (
                1
                / n_samples
                * (probabilities[:, :number_of_weights_row] - free_targets).T
                @ Z
            )
            if np.max(np.abs(gradient)) <= self.tolerance:
                self.converged_ = True
                break
            hessian = np.empty(
                (number_of_weights_row * m_columns, number_of_weights_row * m_columns)
            )
            for k in range(number_of_weights_row):
                row = slice(k * m_columns, (k + 1) * m_columns)

                for j in range(number_of_weights_row):
                    column = slice(j * m_columns, (j + 1) * m_columns)

                    sample_weights = probabilities[:, k] * (
                        float(k == j) - probabilities[:, j]
                    )
                    hessian[row, column] = (
                        Z.T @ (sample_weights[:, None] * Z) / n_samples
                    )
            hessian += self.damping * np.eye(hessian.shape[0])

            step = np.linalg.solve(hessian, gradient.ravel()).reshape(weights.shape)

            directional_decrease = np.sum(gradient * step)
            if (
                not np.isfinite(step).all()
                or not np.isfinite(directional_decrease)
                or directional_decrease <= 0
            ):
                raise RuntimeError("Could not compute a descent direction.")

            current_loss = self.loss_history_[-1]
            step_size = 1.0

            for _ in range(50):
                candidate = weights - step_size * step
                candidate_loss = self._loss(Z, index, candidate)

                if (
                    np.isfinite(candidate_loss)
                    and candidate_loss
                    <= current_loss - 1e-4 * step_size * directional_decrease
                ):
                    break

                step_size *= 0.5
            else:
                raise RuntimeError(
                    "Line search failed. Consider standardizing features "
                    "or increasing damping."
                )

            weights = candidate
            self.loss_history_.append(float(candidate_loss))
            self.n_iterations_ += 1

        # Check convergence after the final accepted update as well.
        final_probabilities = softmax(self._scores(Z, weights), axis=1)
        final_gradient = (
            (final_probabilities[:, :number_of_weights_row] - free_targets).T
            @ Z
            / n_samples
        )
        self.converged_ = np.max(np.abs(final_gradient)) <= self.tolerance

        all_weights = np.vstack([weights, np.zeros((1, m_columns))])

        if self.fit_intercept:
            self.intercept_ = all_weights[:, 0].copy()
            self.coef_ = all_weights[:, 1:].copy()
        else:
            self.intercept_ = np.zeros(self.k_classes)
            self.coef_ = all_weights.copy()

        return self

    def decision_function(self, features):
        """Return the linear score for each class."""
        if self.coef_ is None:
            raise RuntimeError("fit must be called before prediction")
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.m_features:
            raise ValueError(f"features must have shape (n_samples, {self.m_features})")
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values.")
        return np.dot(features, self.coef_.T) + self.intercept_

    def predict_proba(self, features):
        """Compute conditional class probabilities.

        Parameters
        ----------
        features : array-like of shape (n_samples, m_features)
            Input features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, k_classes)
            Class probabilities in ``classes_`` order.
        """
        return softmax(self.decision_function(features), axis=1)

    def predict(self, features):
        """Predict the most probable class.

        Parameters
        ----------
        features : array-like of shape (n_samples, m_features)
            Input features.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted original class labels.
        """
        indices = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[indices]

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
    def _scores(Z, weights):
        """Compute scores with a zero-score reference class."""
        return np.column_stack(
            [
                Z @ weights.T,
                np.zeros(Z.shape[0]),
            ]
        )

    @classmethod
    def _loss(cls, Z, indice, weights):
        """Compute stable mean negative log-likelihood."""
        scores = cls._scores(Z, weights)
        return np.mean(
            logsumexp(scores, axis=1) - scores[np.arange(Z.shape[0]), indice]
        )
