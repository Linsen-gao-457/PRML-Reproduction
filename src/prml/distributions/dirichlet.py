"""Dirichlet distribution over categorical probability vectors."""

import numpy as np
from scipy.special import gamma

from .rv import RandonVariable


class Dirichlet(RandonVariable):
    """Represent a Dirichlet distribution.

    Parameters
    ----------
    alpha : array_like of shape (n_classes,)
        Positive concentration parameters for at least two categories.

    Attributes
    ----------
    size : int
        Number of categories.
    mean : numpy.ndarray of shape (n_classes,)
        Mean probability vector.
    """

    def __init__(self, alpha):
        """Initialize the concentration vector."""
        super().__init__()
        self.alpha = alpha

    @property
    def alpha(self):
        """Return the concentration vector."""
        return self.parameters["alpha"]

    @property
    def size(self):
        """Return the number of categories."""
        return self.alpha.size

    @property
    def mean(self):
        """Return the mean probability vector."""
        return self.alpha / self.alpha.sum()

    @alpha.setter
    def alpha(self, value):
        """Set the positive concentration vector."""
        value = np.asarray(value, dtype=float)
        if value.ndim != 1:
            raise ValueError("alpha must contain at least two values")
        if value.size < 2:
            raise ValueError("alpha must contain at least 2 values")
        if np.any(value <= 0):
            raise ValueError("alpha must be positive")
        if not np.all(np.isfinite(value)):
            raise ValueError("alpha cannot contain infinite values")
        self.parameters["alpha"] = value.copy()

    def _pdf(self, mu):
        """Evaluate density for probability vectors.

        Parameters
        ----------
        mu : array_like of shape (n_classes,) or (n_samples, n_classes)
            Probability vectors on the simplex.

        Returns
        -------
        float or numpy.ndarray of shape (n_samples,)
            Density for each probability vector.
        """
        mu = np.asarray(mu)
        single_point = mu.ndim == 1
        mu = self._valid_mu(mu)
        mu = mu.reshape(-1, self.size)
        normalizer = gamma(self.alpha.sum()) / np.prod(gamma(self.alpha))
        density = normalizer * np.prod(mu ** (self.alpha - 1), axis=1)
        if single_point:
            return density[0]
        return density

    def _draw(self, sample_size, rng):
        """Draw probability vectors from the Dirichlet distribution."""
        return rng.dirichlet(
            self.alpha,
            size=sample_size,
        )

    def _valid_mu(self, mu):
        """Validate probability vectors.

        Parameters
        ----------
        mu : array_like
            Candidate probability vectors.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Validated probability vectors.
        """
        mu = np.asarray(mu, dtype=float)
        if mu.ndim == 1:
            mu = mu.reshape(1, -1)
        if mu.ndim != 2:
            raise ValueError("mu must have shape (n_samples, n_classes)")
        if mu.shape[1] != self.size:
            raise ValueError(f"expected {self.size} dimensions, got {mu.shape[1]}")

        if not np.all(np.isfinite(mu)):
            raise ValueError("mu must contain only finite values")

        if np.any(mu < 0.0):
            raise ValueError("mu must be non-negative")

        if not np.all(
            np.isclose(
                mu.sum(axis=1),
                1.0,
                atol=1e-10,
                rtol=0.0,
            )
        ):
            raise ValueError("each mu must sum to 1")

        return mu
