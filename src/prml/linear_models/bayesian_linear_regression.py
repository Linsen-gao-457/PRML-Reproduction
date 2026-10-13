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
    def precision(self):
        return self._prior

    @precision.setter
    def precision(self, value):
        if not np.isscalar(value):
            raise TypeError("noise_precision must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("noise_precision must be finite")

        if value <= 0:
            raise ValueError("noise_precision must be positive")

        self._precision = value

    @property
    def posterior_mean(self):
        return self.posterior.mean

    @property
    def posterior_covariance(self):
        return self.posterior.cov

    def udpate(basis, target):
        pass

    def sample_weight(self, sample_size, rng):
        return self.posterior.draw(
            sample_size=sample_size,
            rng=rng,
        )

    def _validate_training_data(
        self,
        design_matrix,
        targets,
    ):
        design_matrix = np.asarray(
            design_matrix,
            dtype=float,
        )
        targets = np.asarray(
            targets,
            dtype=float,
        )

        if design_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")

        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")

        if design_matrix.shape[0] != targets.size:
            raise ValueError(
                "design_matrix and targets must contain the same number of observations"
            )

        if design_matrix.shape[1] != self.prior.mean.size:
            raise ValueError("design_matrix has the wrong number of features")

        if not np.all(np.isfinite(design_matrix)):
            raise ValueError("design_matrix must contain only finite values")

        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must contain only finite values")

        return design_matrix, targets
