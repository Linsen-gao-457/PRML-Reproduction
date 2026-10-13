import numpy as np

from prml.distributions.multivariate_gaussian import MultivariateGaussian


class BayesianLinearRegression:
    def __init__(
        self,
        prior,
        precision,
    ):
        self.prior = prior
        self.precision = precision
        self.posterior = MultivariateGaussian(
            mean=self.prior.mean.copy(), cov=self.prior.cov.copy()
        )

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        if not isinstance(value, MultivariateGaussian):
            raise TypeError("prior must be a MultivariateGaussian")

        if value.mean is None or value.cov is None:
            raise ValueError("prior mean and covariance must be known")

        self._prior = value

    @property
    def precision():
        pass

    def udpate(basis, target):
        pass

    def sample_weight(sample_size, rng):
        pass
