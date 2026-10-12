"""Normal-gamma distribution for Gaussian mean and precision."""

import numpy as np
from scipy.special import gamma

from .rv import RandonVariable


class NormalGamma(RandonVariable):
    """Represent a constrained normal-gamma distribution.

    Parameters
    ----------
    beta : float
        Positive conditional precision scale.
    mu_0 : float
        Location of the conditional Gaussian distribution.
    a : float
        Positive gamma shape satisfying ``a = 1 + beta / 2``.
    b : float
        Positive gamma rate.
    """

    def __init__(self, beta, mu_0, a, b):
        """Initialize location, scale, shape, and rate parameters."""
        super().__init__()
        self.mu_0 = mu_0
        self.beta = beta
        self.a = a
        self.b = b

    @classmethod
    def form_constraints(cls, c, beta, d):
        """Construct a distribution from constrained parameters.

        Parameters
        ----------
        c : float
            Natural parameter satisfying ``mu_0 = c / beta``.
        beta : float
            Positive conditional precision scale.
        d : float
            Natural parameter satisfying ``b = d - c**2 / (2 * beta)``.

        Returns
        -------
        NormalGamma
            Distribution expressed by the equivalent standard parameters.
        """
        if not np.isscalar(c) or not np.isscalar(beta) or not np.isscalar(d):
            raise TypeError("c, beta, and d must be scalars")
        c, beta, d = float(c), float(beta), float(d)
        if not np.all(np.isfinite([c, beta, d])):
            raise ValueError("c, beta, and d must be finite")
        if beta <= 0:
            raise ValueError("beta must be positive")
        mu_0 = c / beta
        a = 1 + beta / 2
        b = d - c**2 / (2 * beta)
        if b <= 0:
            raise ValueError("d must be greater than c**2 / (2 * beta)")
        return cls(beta=beta, mu_0=mu_0, a=a, b=b)

    @property
    def mu_0(self):
        """Return the conditional Gaussian location."""
        return self.parameters["mu_0"]

    @mu_0.setter
    def mu_0(self, value):
        """Set the finite conditional Gaussian location."""
        if not np.isscalar(value):
            raise TypeError("mu_0 must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("mu_0 must be finite")

        self.parameters["mu_0"] = value

    @property
    def beta(self):
        """Return the conditional precision scale."""
        return self.parameters["beta"]

    @beta.setter
    def beta(self, value):
        """Set the positive conditional precision scale."""
        if not np.isscalar(value):
            raise TypeError("beta must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("beta must be finite")

        if value <= 0:
            raise ValueError("beta must be positive")

        self.parameters["beta"] = value

    @property
    def a(self):
        """Return the gamma shape parameter."""
        return self.parameters["a"]

    @a.setter
    def a(self, value):
        """Set the shape subject to the normal-gamma constraint."""
        if not np.isscalar(value):
            raise TypeError("a must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("a must be finite")

        if value <= 0:
            raise ValueError("a must be positive")
        expected_a = 1.0 + self.beta / 2.0

        if not np.isclose(value, expected_a):
            raise ValueError(f"constraint requires a = 1 + beta / 2 = {expected_a}")

        self.parameters["a"] = value

    @property
    def b(self):
        """Return the gamma rate parameter."""
        return self.parameters["b"]

    @b.setter
    def b(self, value):
        """Set the positive gamma rate parameter."""
        if not np.isscalar(value):
            raise TypeError("b must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("b must be finite")

        if value <= 0:
            raise ValueError("b must be positive")

        self.parameters["b"] = value

    @property
    def c(self):
        """Return the constrained natural parameter ``c``."""
        return self.beta * self.mu_0

    @property
    def d(self):
        """Return the constrained natural parameter ``d``."""
        return self.b + self.c**2 / (2.0 * self.beta)

    def _pdf(self, x):
        """Evaluate joint density for mean-precision pairs.

        Parameters
        ----------
        x : array_like of shape (2,) or (n_samples, 2)
            Pairs whose first column is mean and second column is precision.

        Returns
        -------
        float or numpy.ndarray of shape (n_samples,)
            Joint density for each pair.
        """
        x = np.asarray(x, dtype=float)
        single_point_sign = x.ndim == 1
        if single_point_sign:
            if x.shape != (2,):
                raise ValueError("single point must have shape (2, )")
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("value must have shape (N,2)")

        mu = x[:, 0]
        precision = x[:, 1]
        density = np.zeros(x.shape[0], dtype=float)
        valid = precision > 0
        precision = precision[valid]
        gamma_density = (
            self.b**self.a
            / gamma(self.a)
            * precision ** (self.a - 1)
            * np.exp(-self.b * precision)
        )
        gaussian_density = np.sqrt(self.beta * precision / (2 * np.pi)) * np.exp(
            -0.5 * self.beta * precision * (mu[valid] - self.mu_0) ** 2
        )
        density[valid] = gamma_density * gaussian_density
        return density[0] if single_point_sign else density

    def _draw(self, sample_size, rng=None):
        """Draw joint mean-precision samples.

        Parameters
        ----------
        sample_size : int
            Number of samples.
        rng : numpy.random.Generator or None, default=None
            Generator to use.

        Returns
        -------
        numpy.ndarray of shape (sample_size, 2)
            Mean-precision pairs.
        """
        if rng is None:
            rng = np.random.default_rng()
        precision = rng.gamma(shape=self.a, scale=1 / self.b, size=sample_size)
        gaussian_variance = 1 / (self.beta * precision)
        mean = rng.normal(
            loc=self.mu_0, scale=np.sqrt(gaussian_variance), size=sample_size
        )
        return np.column_stack((mean, precision))
