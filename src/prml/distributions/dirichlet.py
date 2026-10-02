import numpy as np
from scipy.special import gamma

from .rv import RandonVariable


class Dirichlet(RandonVariable):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    @property
    def alpha(self):
        return self.parameters["alpha"]

    @property
    def size(self):
        return self.alpha.size

    @property
    def mean(self):
        return self.alpha / self.alpha.sum()

    @alpha.setter
    def alpha(self, value):
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
        return rng.dirichlet(
            self.alpha,
            size=sample_size,
        )

    def _valid_mu(self, mu):
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
