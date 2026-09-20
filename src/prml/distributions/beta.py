from math import lgamma

import numpy as np

from .rv import RandonVariable


class Beta(RandonVariable):
    def __init__(self, a, b):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.parameters = {
            "a": self.a,
            "b": self.b,
        }

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    @property
    def variance(self):
        sum = self.a + self.b
        return (self.a * self.b) / (sum**2 * (sum + 1))

    # Heavey engineer implement
    def _pdf(self, x):
        scalar_input = x.ndim == 0
        values = x.reshape(-1)
        density = np.zeros_like(values)

        interior = (values > 0.0) & (values < 1.0)

        log_normalizer = lgamma(self.a + self.b) - lgamma(self.a) - lgamma(self.b)

        density[interior] = np.exp(
            log_normalizer
            + (self.a - 1.0) * np.log(values[interior])
            + (self.b - 1.0) * np.log1p(-values[interior])
        )

        at_zero = values == 0.0
        if self.a < 1.0:
            density[at_zero] = np.inf
        elif self.a == 1.0:
            density[at_zero] = self.b

        at_one = values == 1.0
        if self.b < 1.0:
            density[at_one] = np.inf
        elif self.b == 1.0:
            density[at_one] = self.a

        density[np.isnan(values)] = np.nan

        if scalar_input:
            return density[0]

        return density.reshape(x.shape)

    def _draw(self, sample_size, rng):
        return rng.beta(self.a, self.b, size=sample_size)
