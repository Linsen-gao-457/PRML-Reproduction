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
        return self._precision

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

    def fit(self, design_matrix, targets):
        design_matrix, targets = self._validate_training_data(
            design_matrix,
            targets,
        )
        posterior_precision = (
            self.prior.inv_cov + self.precision * design_matrix.T @ design_matrix
        )
        posterior_covariance = np.linalg.solve(
            posterior_precision,
            np.eye(self.prior.mean.size),
        )
        posterior_natural_mean = (
            self.prior.inv_cov @ self.prior.mean
            + self.precision * design_matrix.T @ targets
        )
        posterior_mean = posterior_covariance @ posterior_natural_mean
        self.posterior = MultivariateGaussian(
            mean=posterior_mean,
            cov=posterior_covariance,
        )

        return self

    def update(self, basis, target):
        basis = np.asarray(
            basis,
            dtype=float,
        )

        if basis.shape != self.posterior.mean.shape:
            raise ValueError("basis must have shape (n_features,)")

        if not np.all(np.isfinite(basis)):
            raise ValueError("basis must contain only finite values")

        if not np.isscalar(target):
            raise TypeError("target must be a scalar")

        target = float(target)
        if not np.isfinite(target):
            raise ValueError("target must be finite")
        current_precision = self.posterior.inv_cov
        current_natural_mean = current_precision @ self.posterior.mean
        updated_precision = current_precision + self.precision * np.outer(basis, basis)
        updated_natural_mean = current_natural_mean + self.precision * basis * target
        identity = np.eye(basis.size)

        updated_covariance = np.linalg.solve(
            updated_precision,
            identity,
        )
        updated_mean = np.linalg.solve(
            updated_precision,
            updated_natural_mean,
        )
        self.posterior = MultivariateGaussian(
            mean=updated_mean,
            cov=updated_covariance,
        )

        return self

    def predict(self, design_matrix):
        design_matrix = np.asarray(
            design_matrix,
            dtype=float,
        )

        if design_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")

        if design_matrix.shape[1] != self.posterior.mean.size:
            raise ValueError("design_matrix has the wrong number of features")

        if not np.all(np.isfinite(design_matrix)):
            raise ValueError("design_matrix must contain only finite values")

        return np.dot(design_matrix, self.posterior.mean)

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
