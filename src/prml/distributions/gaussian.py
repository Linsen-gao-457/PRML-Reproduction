"""Univariate Gaussian distributions and conjugate Bayesian updates."""

import numpy as np

from .gamma import Gamma
from .normal_gamma import NormalGamma
from .rv import RandonVariable


class Gaussian(RandonVariable):
    """Represent or fit a univariate Gaussian distribution.

    Parameters
    ----------
    mean : float or None, default=None
        Distribution mean, or ``None`` before fitting.
    standard_deviation : float or None, default=None
        Positive standard deviation, or ``None`` before fitting.
    """

    def __init__(self, mean=None, standard_deviation=None):
        """Initialize known parameters or fitting placeholders."""
        super().__init__()
        self.mean = mean
        self.standard_deviation = standard_deviation

    @property
    def mean(self):
        """Return the Gaussian mean."""
        return self.parameters["mean"]

    @mean.setter
    def mean(self, value):
        """Set the Gaussian mean."""
        if value == None:
            self.parameters["mean"] = None
            return
        if not np.isscalar(value):
            raise TypeError("value msut be a scalar")
        if not np.isfinite(value):
            raise ValueError("value cannot be infinite")
        self.parameters["mean"] = float(value)

    @property
    def standard_deviation(self):
        """Return the Gaussian standard deviation."""
        return self.parameters["standard_deviation"]

    @standard_deviation.setter
    def standard_deviation(self, value):
        """Set the positive Gaussian standard deviation."""
        if value == None:
            self.parameters["standard_deviation"] = None
            return
        if not np.isscalar(value):
            raise TypeError("value must be scalar")
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if value <= 0:
            raise ValueError("standard_deviation cannot be non-positive")
        self.parameters["standard_deviation"] = float(value)

    def _pdf(self, x):
        """Evaluate the Gaussian density.

        Parameters
        ----------
        x : numpy.ndarray
            Evaluation points.

        Returns
        -------
        numpy.ndarray
            Density at each point.
        """
        if self.mean == None or self.standard_deviation == None:
            raise RuntimeError("gaussian parameters are unknow; call fit first")
        normalization = 1 / (np.sqrt(2 * np.pi * self.standard_deviation**2))
        exponent = np.exp(-1 / (2 * self.standard_deviation**2) * (x - self.mean) ** 2)
        density = normalization * exponent
        return density

    def _draw(self, sample_size, rng):
        """Draw Gaussian samples."""
        if self.mean == None or self.standard_deviation == None:
            raise RuntimeError("gaussian parameters are unknow; call fit first")
        return rng.normal(self.mean, self.standard_deviation, sample_size)

    def fit(self, observartions):
        """Estimate mean and sample standard deviation.

        Parameters
        ----------
        observartions : array_like of shape (n_samples,)
            At least two finite observations.

        Returns
        -------
        Gaussian
            Fitted distribution.
        """
        observartions = self._validate_observations(observations=observartions)
        self.mean = np.mean(observartions)
        fitted_varaince = np.var(observartions, ddof=1)
        self.standard_deviation = np.sqrt(fitted_varaince)
        return self

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
        observations = np.asarray(observations, dtype=float)

        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional")

        if observations.size < 2:
            raise ValueError("at least two observations are required")

        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must contain only finite values")

        return observations


class GaussianMeanBayes:
    """Infer a Gaussian mean with known observation precision.

    Parameters
    ----------
    prior : Gaussian
        Gaussian prior over the unknown mean.
    precision : float
        Positive known precision of the observations.
    """

    def __init__(self, prior, precision):
        """Initialize the prior and known precision."""
        self.prior = prior
        self.precision = precision

    @property
    def prior(self):
        """Return the Gaussian prior."""
        return self._prior

    @prior.setter
    def prior(self, value):
        """Set the Gaussian prior."""
        if not isinstance(value, Gaussian):
            raise TypeError("prior must be Gaussian")
        self._prior = value

    @property
    def precision(self):
        """Return the known observation precision."""
        return self._precision

    @precision.setter
    def precision(self, value):
        """Set the positive known precision."""
        if not np.isscalar(value):
            raise TypeError("precision must be scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("precision must be finite")

        if value <= 0:
            raise ValueError("precision must be positive")

        self._precision = value

    def fit(self, observations):
        """Compute the conjugate posterior over the mean.

        Parameters
        ----------
        observations : array_like of shape (n_samples,)
            Nonempty finite observations.

        Returns
        -------
        Gaussian
            Gaussian posterior over the unknown mean.
        """
        observations = self._validate_obserbations(observations)
        n = observations.size
        mu_ml = observations.mean()
        prior_precision = 1 / (self.prior.standard_deviation**2)
        posterior_precision = prior_precision + n * self.precision
        posterior_standard_deviation = np.sqrt(1.0 / posterior_precision)
        posterior_mu = (
            prior_precision * self.prior.mean + n * self.precision * mu_ml
        ) / posterior_precision
        return Gaussian(
            mean=posterior_mu,
            standard_deviation=posterior_standard_deviation,
        )

    @staticmethod
    def _validate_obserbations(observations):
        """Validate a nonempty univariate sample."""
        try:
            observations = np.asarray(observations, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("observations must be numeric") from error
        if observations.ndim != 1:
            raise TypeError("observations must be one-dimensional")
        if observations.size == 0:
            raise ValueError("observation cannot be empty")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must contain only finite values")
        return observations


class GaussianPrecisionBayes:
    """Infer Gaussian precision with a known mean.

    Parameters
    ----------
    mean : float
        Known observation mean.
    prior : Gamma
        Gamma prior over the unknown precision.
    """

    def __init__(self, mean, prior):
        """Initialize the known mean and gamma prior."""
        self.mean = mean
        self.prior = prior

    @property
    def mean(self):
        """Return the known mean."""
        return self._mean

    @mean.setter
    def mean(self, value):
        """Set the finite known mean."""
        if not np.isscalar(value):
            raise TypeError("mean must be a scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("mean must be finite")
        self._mean = value

    @property
    def prior(self):
        """Return the gamma prior."""
        return self._prior

    @prior.setter
    def prior(self, value):
        """Set the gamma prior."""
        if not isinstance(value, Gamma):
            raise TypeError("prior must follow Gamma distribution")
        self._prior = value

    def fit(self, observations):
        """Compute the conjugate posterior over precision.

        Parameters
        ----------
        observations : array_like of shape (n_samples,)
            Nonempty finite observations.

        Returns
        -------
        Gamma
            Gamma posterior over precision.
        """
        observations = self._validate_observations(observations)
        n = observations.size
        square_error = np.sum((observations - self.mean) ** 2)
        a = self.prior.a + n / 2
        b = self.prior.b + 0.5 * square_error
        return Gamma(a=a, b=b)

    @staticmethod
    def _validate_observations(observations):
        """Validate a nonempty univariate sample."""
        try:
            observations = np.asarray(
                observations,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("observations must contain numeric values") from exc

        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional")

        if observations.size == 0:
            raise ValueError("observations cannot be empty")

        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must contain only finite values")

        return observations


class GaussianMeanPrecisionBayes:
    """Infer an unknown Gaussian mean and precision.

    Parameters
    ----------
    prior : NormalGamma
        Conjugate prior over mean and precision.
    """

    def __init__(self, prior):
        """Initialize the normal-gamma prior."""
        self.prior = prior

    @property
    def prior(self):
        """Return the normal-gamma prior."""
        return self._prior

    @prior.setter
    def prior(self, value):
        """Set the normal-gamma prior."""
        if not isinstance(value, NormalGamma):
            raise TypeError("prior must be NormalGamma")
        self._prior = value

    def fit(self, observations):
        """Compute the conjugate posterior over mean and precision.

        Parameters
        ----------
        observations : array_like of shape (n_samples,)
            Nonempty finite observations.

        Returns
        -------
        NormalGamma
            Normal-gamma posterior.
        """
        observations = self._validate_observations(observations)

        posterior_c = self.prior.c + observations.sum()
        posterior_beta = self.prior.beta + observations.size
        posterior_d = self.prior.d + 0.5 * np.dot(observations, observations)

        return NormalGamma.form_constraints(
            c=posterior_c,
            beta=posterior_beta,
            d=posterior_d,
        )

    @staticmethod
    def _validate_observations(observations):
        """Validate a nonempty univariate sample."""
        try:
            observations = np.asarray(observations, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("observations must contain numeric values") from exc
        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional")
        if observations.size == 0:
            raise ValueError("observations cannot be empty")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must contain only finite values")
        return observations
