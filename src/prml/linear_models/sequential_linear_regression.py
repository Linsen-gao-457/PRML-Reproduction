"""Sequential least-mean-squares linear regression."""

import numpy as np


class SequentialLinearRegression:
    """Learn linear coefficients through per-observation LMS updates.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Positive step size used for every update.
    epochs : int, default=1000
        Positive number of passes over the training observations.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,) or None
        Current coefficients, or ``None`` before fitting.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        """Initialize the learning rate and number of epochs."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.coef_ = None

    def partial_fit(self, phi_x, t):
        """Update coefficients from one observation.

        Parameters
        ----------
        phi_x : array_like of shape (n_features,)
            Basis-function values for one observation.
        t : float
            Target for the observation.
        """
        phi_x = np.asarray(phi_x, dtype=float)
        t = np.asarray(t, dtype=float)
        prediction = np.dot(phi_x, self.coef_)
        error = t - prediction
        self.coef_ += self.learning_rate * error * phi_x

    def fit(self, design_matrix, targets):
        """Run sequential updates for each training epoch.

        Parameters
        ----------
        design_matrix : array_like of shape (n_samples, n_features)
            Basis-function values for the training observations.
        targets : array_like of shape (n_samples,)
            Continuous training targets.

        Returns
        -------
        SequentialLinearRegression
            Fitted estimator.
        """
        design_matrix = np.asanyarray(design_matrix, dtype=float)
        targets = np.asanyarray(targets, dtype=float)
        if design_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")

        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")

        if design_matrix.shape[0] != targets.size:
            raise ValueError(
                "design_matrix and targets must have the same number of observations"
            )
        if not np.all(np.isfinite(design_matrix)):
            raise ValueError("design_matrix must contain finite values")

        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must contain finite values")
        self.coef_ = np.zeros(design_matrix.shape[1], dtype=float)
        for _ in range(self.epochs):
            for phi_x, t in zip(design_matrix, targets):
                self.partial_fit(phi_x, t)
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
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")

        design_matrix = np.asarray(
            design_matrix,
            dtype=float,
        )
        return np.dot(
            design_matrix,
            self.coef_,
        )
