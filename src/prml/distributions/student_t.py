"""Univariate Student's t distribution with EM-style fitting."""

import numpy as np
from scipy.special import digamma, gammaln

from .rv import RandonVariable


class Student_T(RandonVariable):
    """Represent, sample, and fit a Student's t distribution.

    Parameters
    ----------
    mean : float, default=0.0
        Location parameter.
    precision : float, default=1.0
        Positive precision parameter.
    degree_of_freedom : float, default=1.0
        Positive degrees-of-freedom parameter.
    """

    def __init__(self, mean=0.0, precision=1.0, degree_of_freedom=1.0):
        """Initialize location, precision, and degrees of freedom."""
        super().__init__()
        self.mean = mean
        self.precision = precision
        self.degree_of_freedom = degree_of_freedom

    @property
    def mean(self):
        """Return the location parameter."""
        return self.parameters["mean"]

    @mean.setter
    def mean(self, value):
        """Set the finite location parameter."""
        if not np.isscalar(value):
            raise TypeError("mean must be a scalar")

        value = float(value)
        if not np.isfinite(value):
            raise ValueError("mean must be finite")

        self.parameters["mean"] = value

    @property
    def precision(self):
        """Return the precision parameter."""
        return self.parameters["precision"]

    @precision.setter
    def precision(self, value):
        """Set the positive precision parameter."""
        if not np.isscalar(value):
            raise TypeError("precision must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("precision must be finite")

        if value <= 0:
            raise ValueError("precision must be positive")

        self.parameters["precision"] = value

    @property
    def degree_of_freedom(self):
        """Return the degrees-of-freedom parameter."""
        return self.parameters["degree_of_freedom"]

    @degree_of_freedom.setter
    def degree_of_freedom(self, value):
        """Set the positive degrees-of-freedom parameter."""
        if not np.isscalar(value):
            raise TypeError("degree_of_freedom must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("degree_of_freedom must be finite")

        if value <= 0:
            raise ValueError("degree_of_freedom must be positive")

        self.parameters["degree_of_freedom"] = value

    def _pdf(self, x):
        """Evaluate the Student's t density.

        Parameters
        ----------
        x : numpy.ndarray
            Evaluation points.

        Returns
        -------
        float or numpy.ndarray
            Density at each point.
        """
        nu = self.degree_of_freedom
        squared_error = (x - self.mean) ** 2
        log_normalization = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2.0)
            + 0.5 * (np.log(self.precision) - np.log(np.pi * nu))
        )
        log_kernel = -(nu + 1) / 2 * np.log1p(self.precision * squared_error / nu)
        density = np.exp(log_kernel + log_normalization)
        return density.item() if density.ndim == 0 else density

    def _draw(self, sample_size, rng):
        """Draw Student's t samples."""
        scale = 1.0 / np.sqrt(self.precision)
        return self.mean + scale * rng.standard_t(
            df=self.degree_of_freedom,
            size=sample_size,
        )

    def fit(
        self,
        observations,
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-6,
    ):
        """Estimate parameters with latent-scale expectation maximization.

        Parameters
        ----------
        observations : array_like of shape (n_samples,)
            At least two finite observations with positive variance.
        learning_rate : float, default=0.01
            Step size for the degrees-of-freedom update.
        max_iterations : int, default=1000
            Maximum number of EM iterations.
        tolerance : float, default=1e-6
            Relative and absolute parameter convergence tolerance.

        Returns
        -------
        Student_T
            Fitted distribution.

        Raises
        ------
        RuntimeError
            If fitting does not converge or degrees of freedom becomes invalid.
        """
        observations = self._validate_observations(observations)

        variance = np.var(observations)

        if variance <= 0:
            raise ValueError("observations must have positive variance")

        self.mean = np.mean(observations)
        self.precision = 1.0 / variance
        self.degree_of_freedom = 1.0

        parameters = self._pack_parameters()

        for _ in range(max_iterations):
            expected_eta, expected_log_eta = self._expectation(observations)

            self._maximization(
                observations=observations,
                expected_eta=expected_eta,
                expected_log_eta=expected_log_eta,
                learning_rate=learning_rate,
            )

            new_parameters = self._pack_parameters()

            if np.allclose(
                parameters,
                new_parameters,
                rtol=tolerance,
                atol=tolerance,
            ):
                return self

            parameters = new_parameters

        raise RuntimeError("Student-t fitting did not converge")

    def _expectation(self, observations):
        """Compute expected latent scales.

        Parameters
        ----------
        observations : numpy.ndarray of shape (n_samples,)
            Validated observations.

        Returns
        -------
        expected_eta : numpy.ndarray of shape (n_samples,)
            Expected latent scales.
        expected_log_eta : numpy.ndarray of shape (n_samples,)
            Expected logarithms of latent scales.
        """
        nu = self.degree_of_freedom

        squared_error = (observations - self.mean) ** 2

        denominator = nu + self.precision * squared_error

        expected_eta = (nu + 1.0) / denominator

        expected_log_eta = digamma((nu + 1.0) / 2.0) - np.log(denominator / 2.0)

        return expected_eta, expected_log_eta

    def _maximization(
        self,
        observations,
        expected_eta,
        expected_log_eta,
        learning_rate,
    ):
        """Update all distribution parameters for one EM iteration."""
        new_mean = np.sum(expected_eta * observations) / np.sum(expected_eta)

        weighted_squared_error = np.mean(expected_eta * (observations - new_mean) ** 2)

        if weighted_squared_error <= 0:
            raise ValueError("weighted variance must be positive")

        new_precision = 1.0 / weighted_squared_error

        nu = self.degree_of_freedom

        degree_gradient = 0.5 * (
            np.log(nu / 2.0)
            + 1.0
            - digamma(nu / 2.0)
            + np.mean(expected_log_eta - expected_eta)
        )

        new_degree_of_freedom = nu + learning_rate * degree_gradient

        if new_degree_of_freedom <= 0 or not np.isfinite(new_degree_of_freedom):
            raise RuntimeError(
                "degree_of_freedom became invalid; try a smaller learning_rate"
            )

        self.mean = new_mean
        self.precision = new_precision
        self.degree_of_freedom = new_degree_of_freedom

    def _pack_parameters(self):
        """Pack current parameters for convergence testing.

        Returns
        -------
        numpy.ndarray of shape (3,)
            Mean, precision, and degrees of freedom.
        """
        return np.array(
            [
                self.mean,
                self.precision,
                self.degree_of_freedom,
            ]
        )

    @staticmethod
    def _validate_observations(observations):
        """Validate a univariate sample.

        Parameters
        ----------
        observations : array_like
            Candidate observations.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Finite floating-point observations.
        """
        try:
            observations = np.asarray(
                observations,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("observations must contain numeric values") from exc

        if observations.ndim != 1:
            raise ValueError("observations must have shape (N,)")

        if observations.size < 2:
            raise ValueError("at least two observations are required")

        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must contain only finite values")

        return observations
