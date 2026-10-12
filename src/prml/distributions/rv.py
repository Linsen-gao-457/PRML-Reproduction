"""Abstract interface shared by probability distributions."""

from abc import ABC, abstractmethod

import numpy as np


class RandonVariable(ABC):
    """Define the common density and sampling interface.

    Attributes
    ----------
    parameters : dict
        Named parameters used by the concrete distribution.
    """

    def __init__(self):
        """Initialize an empty parameter mapping."""
        self.parameters = {}

    def __repr__(self):
        """Return a constructor-like representation.

        Returns
        -------
        str
            Class name followed by stored parameter representations.
        """
        arguments = ", ".join(
            f"{name}={value!r}" for name, value in self.parameters.items()
        )
        return f"{self.__class__.__name__}({arguments})"

    def pdf(self, x):
        """Evaluate the probability density or mass.

        Parameters
        ----------
        x : array_like
            Evaluation points.

        Returns
        -------
        float or numpy.ndarray
            Density or probability mass at each point.
        """
        x = np.asarray(x, dtype="float")
        return self._pdf(x)

    def draw(self, sample_size=1, rng=None):
        """Draw random samples.

        Parameters
        ----------
        sample_size : int, default=1
            Positive number of samples.
        rng : numpy.random.Generator or None, default=None
            Generator to use. A fresh generator is created when omitted.

        Returns
        -------
        numpy.ndarray
            Random samples in the shape defined by the distribution.
        """
        if not isinstance(sample_size, (int, np.integer)):
            raise TypeError("sample_size must be an integer")
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        rng = rng or np.random.default_rng()
        return self._draw(sample_size, rng)

    @abstractmethod
    def _pdf(self, x):
        """Evaluate a subclass-specific density or mass function."""
        pass

    @abstractmethod
    def _draw(self, sample_size, rng):
        """Draw samples using a subclass-specific implementation."""
        pass
