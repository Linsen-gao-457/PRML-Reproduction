import numpy as np
from scipy.special import gamma

from .rv import RandonVariable


class Gamma(RandonVariable):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    @property
    def a(self):
        return self.parameters["a"]

    @a.setter
    def a(self, value):
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
        return self.parameters["b"]

    @b.setter
    def b(self, value):
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
        return self.a / self.b

    @property
    def variance(self):
        return self.a / (self.b**2)

    def _pdf(self, x):
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
        return super()._draw(sample_size, rng)
