import numpy as np

from .dirichlet import Dirichlet
from .rv import RandonVariable


class Categorical(RandonVariable):
    def __init__(self, mu=None):
        super().__init__()
        self.k_class = None
        self.mu = mu

    @property
    def mu(self):
        return self.parameters["mu"]

    @mu.setter
    def mu(self, mu):
        mu = np.asarray(mu, "float")
        if isinstance(mu, np.ndarray):
            if mu.ndim != 1:
                raise ValueError("dismensionallity of mu must be 1")
            if (mu < 0).any():
                raise ValueError()
            if not np.allclose(mu.sum(), 1):
                raise ValueError("sum of mu must be 1")
            self.k_class = mu.size
            self.mu = mu.copy()

    @property
    def probabilities(self):
        if self.mu is None:
            raise RuntimeError("mu is unknown")
        return self.mu

    @property
    def size(self):
        if hasattr(self.mu, "size"):
            return self.k_class

    @property
    def mean(self):
        return self.probabilities

    def pmf(self, x):
        observations, single_observation_sign = self._validate_observations(
            observations=x
        )
        num_class = observations.sum(axis=0)
        if isinstance(num_class, np.ndarray):
            self.mu = num_class / num_class.sum()
        return self

    def _pdf(self, x):
        return self.pmf(x)

    def _draw(self, sample_size, rng):
        return rng.choice(self.k_class, size=sample_size)

    def _validate_observations(
        self,
        observations,
    ):
        observations = np.asarray(observations)
        single_obervation_sign = observations.ndim == 1
        if single_obervation_sign:
            observations = observations.reshape(1, -1)
        if observations.ndim != 2:
            raise ValueError("observations must have shape (n_samples, k_classes)")
        if observations.shape[0] == 0:
            raise ValueError("Observation canot be empty.")
        if observations.shape[1] < 2:
            raise ValueError("Observations should contain at least two classes.")
        if observations.shape[1] != self.k_class:
            raise ValueError(
                f"expexted {self.k_class} classes,received {observations.shape[1]}"
            )
        if not np.all(observations == 0 | observations == 1):
            raise ValueError("each element of observations should be 0 or 1")
        if not np.all(observations.sum(axis=1) == 1):
            raise ValueError("each obervation must contain one active class")
        return (observations.astype(int), single_obervation_sign)
