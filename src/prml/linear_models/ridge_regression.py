"""Linear regression with an L2 coefficient penalty."""

import numpy as np


class RidgeRegression:
    """Fit ridge regression to a precomputed design matrix.

    Parameters
    ----------
    regularization : float
        Nonnegative L2 penalty applied to every coefficient.

    Attributes
    ----------
    regularization : float
        L2 penalty used during fitting.
    """

    def __init__(self, regularization):
        """Initialize the regularization strength."""
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self.regularization = regularization
        self._coef = None

    def fit(self, design_matrix, targets):
        """Estimate coefficients using the regularized normal equation.

        Parameters
        ----------
        design_matrix : array_like of shape (n_samples, n_features)
            Basis-function values for the training observations.
        targets : array_like of shape (n_samples,)
            Continuous training targets.

        Returns
        -------
        RidgeRegression
            Fitted estimator.
        """
        design_matrix = np.asarray(design_matrix)
        targets = np.asarray(targets)
        identity = np.eye(design_matrix.shape[1])
        lhs = np.dot(design_matrix.T, design_matrix)
        lhs += self.regularization * identity
        rhs = np.dot(design_matrix.T, targets)
        self._coef = np.linalg.solve(lhs, rhs)
        return self

    def predict(self, design_matrix):
        """Predict targets for rows of a design matrix.

        Parameters
        ----------
        design_matrix : array_like of shape (n_samples, n_features)
            Basis-function values for prediction.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted targets.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        """
        if self._coef is None:
            raise RuntimeError("fit must be called before predict")
        return np.dot(design_matrix, self._coef)
