import numpy as np

from prml.distributions.beta import Beta
from prml.distributions.rv import RandonVariable


class Bernoulli(RandonVariable):
    def __init__(self, mu=None):
        super().__init__()
        self.mu = mu

    @property
    def mu(self):
        return self.parameters["mu"]

    @mu.setter
    def mu(self, value):
        if value is None or isinstance(value, Beta):
            self.parameters["mu"] = value
            return
        if not np.isscalar(value):
            raise TypeError("mu must be a scalar, Beta distribution, or None")

        value = float(value)

        if not 0 <= value <= 1:
            raise ValueError("mu must be between 0 and 1")
        self.parameters["mu"] = value

    @property
    def probability(self):
        if self.mu is None:
            raise RuntimeError("mu is unknown; call fit before using the distribution")
        if isinstance(self.mu, Beta):
            return self.mu.mean
        return self.mu

    @property
    def mean(self):
        return self.probability

    @property
    def variance(self):
        mean = self.probability
        return mean * (1 - mean)

    def pmf(self, x):
        x = self._validate_observations(x)
        probability = self.probability
        result = np.where(x == 1, probability, 1.0 - probability)
        return result.item() if result.ndim == 0 else result

    def fit(self, observations):
        observations = self._validate_observations(observations)
        number_of_ones = np.count_nonzero(observations)
        number_of_zeros = observations.size - number_of_ones
        if isinstance(self.mu, Beta):
            self.mu = Beta(a=self.mu.a + number_of_ones, b=self.mu.b + number_of_zeros)
        else:
            self.mu = number_of_ones / observations.size
        return self

    def _pdf(self, x):
        return self.pmf(x)

    def _draw(self, sample_size, rng):
        return rng.binomial(n=1, p=self.probability, size=sample_size)

    @staticmethod
    def _validate_observations(observations):
        observations = np.asarray(observations)

        if observations.size == 0:
            raise ValueError("observations cannot be empty")

        if not np.all((observations == 0) | (observations == 1)):
            raise ValueError("Bernoulli observations must be either 0 or 1")

        return observations.astype(int)
