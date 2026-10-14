"""Bayesian linear regression with a Gaussian weight prior."""

import numpy as np

from prml.distributions.multivariate_gaussian import MultivariateGaussian


class BayesianLinearRegression:
    """Fit a linear basis-function model by Bayesian inference.

    The model assumes Gaussian observation noise with known precision and a
    multivariate Gaussian prior over the weight vector. The posterior is
    available after either a batch fit or one or more sequential updates.

    Parameters
    ----------
    prior : MultivariateGaussian
        Prior distribution over the ``n_features`` model weights. Its mean
        and covariance must both be known.
    precision : float
        Positive observation-noise precision, equal to the reciprocal of the
        observation-noise variance.

    Attributes
    ----------
    posterior : MultivariateGaussian
        Current posterior distribution over the weight vector. It initially
        equals the prior.
    """

    def __init__(
        self,
        prior,
        precision,
    ):
        """Initialize the model from a weight prior and noise precision."""
        self.prior = prior
        self.precision = precision
        self.posterior = MultivariateGaussian(
            mean=self.prior.mean.copy(), cov=self.prior.cov.copy()
        )

    @property
    def prior(self):
        """MultivariateGaussian: Prior distribution over model weights."""
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
        """float: Observation-noise precision."""
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
        """numpy.ndarray of shape (n_features,): Posterior weight mean."""
        return self.posterior.mean

    @property
    def posterior_covariance(self):
        """Return the posterior covariance with shape (n_features, n_features)."""
        return self.posterior.cov

    def fit(self, design_matrix, targets):
        """Compute the posterior from a complete training dataset.

        Parameters
        ----------
        design_matrix : array_like of shape (n_samples, n_features)
            Basis-function values for every training observation.
        targets : array_like of shape (n_samples,)
            Scalar target value for every training observation.

        Returns
        -------
        BayesianLinearRegression
            Fitted model with an updated posterior distribution.

        Raises
        ------
        ValueError
            If the input dimensions are incompatible, the feature count does
            not match the prior, or either input contains a non-finite value.
        """
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
        """Update the posterior using one observation.

        Parameters
        ----------
        basis : array_like of shape (n_features,)
            Basis-function vector for one input point.
        target : float
            Scalar target associated with ``basis``.

        Returns
        -------
        BayesianLinearRegression
            Model with its posterior updated in place.

        Raises
        ------
        TypeError
            If ``target`` is not scalar.
        ValueError
            If ``basis`` has the wrong shape, or if either argument contains
            a non-finite value.
        """
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
        """Return the posterior-mean prediction for each input.

        This method returns only the predictive mean ``Phi @ m_N``. It does
        not include observation noise or posterior parameter uncertainty.

        Parameters
        ----------
        design_matrix : array_like of shape (n_samples, n_features)
            Basis-function values for the inputs to predict.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Prediction from the posterior mean weight vector.

        Raises
        ------
        ValueError
            If the design matrix is not two-dimensional, has the wrong number
            of features, or contains a non-finite value.
        """
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
        """Draw weight vectors from the current posterior.

        Parameters
        ----------
        sample_size : int
            Number of weight vectors to draw.
        rng : numpy.random.Generator
            Random number generator used for sampling.

        Returns
        -------
        numpy.ndarray of shape (sample_size, n_features)
            Independent samples from the posterior weight distribution.
        """
        return self.posterior.draw(
            sample_size=sample_size,
            rng=rng,
        )

    def predictive_distribution(self, design_matrix):
        design_matrix = np.asarray(
            design_matrix,
            dtype=float,
        )

        predictive_mean = np.dot(design_matrix, self.posterior.mean)

        predictive_variance = 1.0 / self.precision + np.einsum(
            "ni,ij,nj->n",
            design_matrix,
            self.posterior.cov,
            design_matrix,
        )

        return predictive_mean, predictive_variance

    def _validate_training_data(
        self,
        design_matrix,
        targets,
    ):
        """Convert and validate batch training inputs.

        Parameters
        ----------
        design_matrix : array_like of shape (n_samples, n_features)
            Basis-function values for every training observation.
        targets : array_like of shape (n_samples,)
            Scalar target value for every training observation.

        Returns
        -------
        design_matrix : numpy.ndarray of shape (n_samples, n_features)
            Validated floating-point design matrix.
        targets : numpy.ndarray of shape (n_samples,)
            Validated floating-point target vector.

        Raises
        ------
        ValueError
            If the input dimensions are incompatible, the feature count does
            not match the prior, or either input contains a non-finite value.
        """
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
