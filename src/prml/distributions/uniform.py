"""Continuous uniform distribution on a finite interval."""

import numpy as np

from .rv import RandonVariable


class Uniform(RandonVariable):
    """Represent a continuous uniform distribution.

    Parameters
    ----------
    low : float, default=0.0
        Finite lower endpoint.
    high : float, default=1.0
        Finite upper endpoint greater than ``low``.
    """

    def __init__(self, low=0.0, high=1.0):
        """Initialize the support interval."""
        super().__init__()
        self.low = low
        self.high = high

    @property
    def low(self):
        """Return the lower endpoint."""
        return self.parameters["low"]

    @low.setter
    def low(self, value):
        """Set the finite lower endpoint."""
        if not np.isscalar(value):
            raise TypeError("low must be scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("low cannot be infinite")
        if "high" in self.parameters and value >= self.high:
            raise ValueError("low must be smaller than high")
        self.parameters["low"] = value

    @property
    def high(self):
        """Return the upper endpoint."""
        return self.parameters["high"]

    @high.setter
    def high(self, value):
        """Set the finite upper endpoint."""
        if not np.isscalar(value):
            raise TypeError("high must be a scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("high must be finite")
        if "low" in self.parameters and value <= self.low:
            raise ValueError("high must be larger than low")
        self.parameters["high"] = value

    def _pdf(self, x):
        """Evaluate the uniform density.

        Parameters
        ----------
        x : numpy.ndarray
            Evaluation points.

        Returns
        -------
        float or numpy.ndarray
            Constant density on the support and zero elsewhere.
        """
        density = 1.0 / (self.high - self.low)
        result = np.where((x >= self.low) & (x <= self.high), density, 0.0)
        return result.item() if result.ndim == 0 else result

    def _draw(self, sample_size, rng):
        """Draw uniformly distributed samples."""
        return rng.uniform(low=self.low, high=self.high, size=sample_size)
