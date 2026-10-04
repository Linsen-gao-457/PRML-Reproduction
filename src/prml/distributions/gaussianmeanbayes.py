import numpy as np

from .gaussian import Gaussian

class GaussianMeanBayes:
    def __init__(self, prior, observation_precision):
    if not isinstance(prior, Gaussian):
        raise TypeError("prior must be a Gaussian")

    if (
        prior.mean is None
        or prior.standard_deviation is None
    ):
        raise ValueError("prior Gaussian must have known parameters")

    if not np.isscalar(observation_precision):
        raise TypeError("observation_precision must be a scalar")

    observation_precision = float(observation_precision)

    if not np.isfinite(observation_precision):
        raise ValueError("observation_precision must be finite")

    if observation_precision <= 0:
        raise ValueError("observation_precision must be positive")

    self.prior = prior
    self.observation_precision = observation_precision

def fit(self, observations):
    observations = np.asarray(observations, dtype=float)

    if observations.ndim != 1:
        raise ValueError("observations must be one-dimensional")

    if observations.size == 0:
        raise ValueError("observations cannot be empty")

    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")

    n = observations.size
    sample_mean = observations.mean()

    prior_precision = (
        1.0 / self.prior.standard_deviation**2
    )

    posterior_precision = (
        prior_precision
        + n * self.observation_precision
    )

    posterior_mean = (
        prior_precision * self.prior.mean
        + n * self.observation_precision * sample_mean
    ) / posterior_precision

    posterior_standard_deviation = np.sqrt(
        1.0 / posterior_precision
    )

    return Gaussian(
        mean=posterior_mean,
        standard_deviation=posterior_standard_deviation,
    )