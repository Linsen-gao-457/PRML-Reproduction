"""Beta distribution on probabilities in the unit interval."""

import numpy as np
from scipy.special import gamma

from .rv import RandonVariable


class Beta(RandonVariable):
    """Represent a beta distribution.

    Parameters
    ----------
    a : float
        First shape parameter.
    b : float
        Second shape parameter.

    Attributes
    ----------
    mean : float
        Distribution mean.
    variance : float
        Distribution variance.
    """

    def __init__(self, a, b):
        """Initialize the shape parameters."""
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.parameters = {
            "a": self.a,
            "b": self.b,
        }

    @property
    def mean(self):
        """Return the distribution mean."""
        return self.a / (self.a + self.b)

    @property
    def variance(self):
        """Return the distribution variance."""
        sum = self.a + self.b
        return (self.a * self.b) / (sum**2 * (sum + 1))

    def _pdf(self, mu):
        """Evaluate the beta density.

        Parameters
        ----------
        mu : numpy.ndarray
            Probability values at which to evaluate the density.

        Returns
        -------
        numpy.ndarray
            Density at each probability value.
        """
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            density = (
                gamma(self.a + self.b)
                * np.power(mu, self.a - 1)
                * np.power(1 - mu, self.b - 1)
                / gamma(self.a)
                / gamma(self.b)
            )

        return density

    def _draw(self, sample_size, rng):
        """Draw beta-distributed samples."""
        return rng.beta(self.a, self.b, size=sample_size)
