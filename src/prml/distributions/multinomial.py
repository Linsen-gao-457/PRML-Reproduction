from math import factorial

import numpy as np

from .rv import RandonVariable


class Multinomial(RandonVariable):
    def __init__(self, n_trials, mu=None):
        super().__init__()
        self.n_trials = self.n_trials
        self.mu = mu

    @property
    def mu(self):
        return self.parameters(["mu"])

    @mu.setter
    def mu(self, value):
        if value is None:
            self.parameters["mu"] = None
            return
        value = np.asarray(value, dtype=float)
        if value.ndim != 1:
            raise ValueError("mu must be one dimensional")
        if np.any(value) < 0:
            raise ValueError("mu must be non-negative")
        if not np.isclose(value.sum(), 1):
            raise ValueError("mu must sum to 1")

        self.parameters["mu"] = value

    def fit(self, observations):
        observations = np.asarray(observations)
        total_counts = observations.sum(axis=0)
        self.mu = total_counts / total_counts.sum()
        return self

    def pmf(self, x):
        x = np.asarray(x, dtype=int)
        coefficient = factorial(self.n_trials)

        for count in x:
            coefficient /= factorial(count)

        return coefficient * np.prod(self.mu**x)

    def pdf(self, x):
        return self.pmf(x)

    def _draw(self, sample_size, rng):
        return rng.multinomial(self.n_trials, self.mu, size=sample_size)
