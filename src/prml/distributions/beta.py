import numpy as np
from scipy.special import gamma

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
    def _pdf(self, mu):
        return (
            gamma(self.a + self.b)
            * np.power(mu, self.a - 1)
            * np.power(1 - mu, self.b - 1)
            / gamma(self.a)
            / gamma(self.b)
        )

    def _draw(self, sample_size, rng):
        return rng.beta(self.a, self.b, size=sample_size)
