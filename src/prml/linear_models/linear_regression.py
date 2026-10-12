"""Ordinary least-squares linear regression."""

import numpy as np


class LinearRegression:
    """Fit a linear model to a precomputed design matrix.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,) or None
        Fitted coefficients, or ``None`` before fitting.
    """

    def __init__(self):
        """Initialize an unfitted estimator."""
        self.coef_ = None

    def fit(self, design_matrix, targets):
        """Estimate coefficients by ordinary least squares.

        Parameters
        ----------
        design_matrix : numpy.ndarray of shape (n_samples, n_features)
            Basis-function values for the training observations.
        targets : numpy.ndarray of shape (n_samples,)
            Continuous training targets.

        Returns
        -------
        LinearRegression
            Fitted estimator.
        """
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
        """Predict targets for rows of a design matrix.

        Parameters
        ----------
        design_matrix : numpy.ndarray of shape (n_samples, n_features)
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
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        if design_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")
        return np.dot(design_matrix, self.coef_)
