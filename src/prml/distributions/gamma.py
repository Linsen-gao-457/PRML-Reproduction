"""Gamma distribution in shape-rate parameterization."""

import numpy as np
from scipy.special import gamma

from .rv import RandonVariable


class Gamma(RandonVariable):
    """Represent a gamma distribution.

    Parameters
    ----------
    a : float
        Positive shape parameter.
    b : float
        Positive rate parameter.

    Attributes
    ----------
    mean : float
        Distribution mean.
    variance : float
        Distribution variance.
    """

    def __init__(self, a, b):
        """Initialize the shape and rate parameters."""
        super().__init__()
        self.a = a
        self.b = b

    @property
    def a(self):
        """Return the shape parameter."""
        return self.parameters["a"]

    @a.setter
    def a(self, value):
        """Set the positive shape parameter."""
        if not np.isscalar(value):
            raise TypeError("a must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("a must be finite")

        if value <= 0:
            raise ValueError("a must be positive")

        self.parameters["a"] = value

    @property
    def b(self):
        """Return the rate parameter."""
        return self.parameters["b"]

    @b.setter
    def b(self, value):
        """Set the positive rate parameter."""
        if not np.isscalar(value):
            raise TypeError("b must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("b must be finite")

        if value <= 0:
            raise ValueError("b must be positive")

        self.parameters["b"] = value

    @property
    def mean(self):
        """Return the distribution mean."""
        return self.a / self.b

    @property
    def variance(self):
        """Return the distribution variance."""
        return self.a / (self.b**2)

    def _pdf(self, x):
        """Evaluate the gamma density.

        Parameters
        ----------
        x : numpy.ndarray
            Evaluation points.

        Returns
        -------
        float or numpy.ndarray
            Density values, with zero density outside the positive support.
        """
        x = np.array(x)
        density = np.zeros_like(x, dtype=float)
        valid = x > 0
        density[valid] = (
            self.b**self.a
            / gamma(self.a)
            * x[valid] ** (self.a - 1)
            * np.exp(-self.b * x[valid])
        )
        return density.item() if density.ndim == 0 else density

    def _draw(self, sample_size, rng):
        """Draw gamma-distributed samples."""
        return rng.gamma(
            shape=self.a,
            scale=1.0 / self.b,
            size=sample_size,
        )
