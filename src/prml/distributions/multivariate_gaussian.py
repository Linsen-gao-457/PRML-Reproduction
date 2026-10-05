import numpy as np

from .rv import RandonVariable


class MultivariateGaussian(RandonVariable):
    def __init__(self, mean=None, cov=None):
        super().__init__()
        self.mean = mean
        self.cov = cov

    @property
    def mean(self):
        return self.parameters["mean"]

    @mean.setter
    def mean(self, value):
        if value is None:
            self.parameters["mean"] = None
            return

        try:
            value = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("mean must contain numeric values") from exc

        if value.ndim != 1:
            raise ValueError("mean must be one-dimensional")

        if value.size == 0:
            raise ValueError("mean cannot be empty")

        if not np.all(np.isfinite(value)):
            raise ValueError("mean must contain only finite values")

        self.parameters["mean"] = value

    @property
    def cov(self):
        return self.parameters["cov"]

    @cov.setter
    def cov(self, value):
        if value is None:
            self.parameters["cov"] = None
            self.parameters["inv_cov"] = None
            return

        try:
            value = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("cov must contain numeric values") from exc

        if value.ndim != 2:
            raise ValueError("cov must be two-dimensional")

        if value.shape[0] != value.shape[1]:
            raise ValueError("cov must be square")

        if value.shape[0] == 0:
            raise ValueError("cov cannot be empty")

        if not np.all(np.isfinite(value)):
            raise ValueError("cov must contain only finite values")

        if not np.allclose(value, value.T):
            raise ValueError("cov must be symmetric")

        if self.mean is not None and value.shape[0] != self.mean.size:
            raise ValueError("cov dimension must match mean dimension")

        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as exc:
            raise ValueError("cov must be positive definite") from exc
        I = np.eye(value.shape[0])
        self.parameters["cov"] = value
        self.parameters["inv_cov"] = np.linalg.solve(value, I)

    @property
    def inv_cov(self):
        return self.parameters["inv_cov"]

    def _pdf(self, x):
        self._check_parameters()
        x = np.asarray(x, dtype=float)
        single_point_sign = x.ndim == 1
        if single_point_sign:
            if x.shape != self.mean.shape:
                raise ValueError(f"a single point must have shape {self.mean.shape}")
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError("x must have shape (N, D)")
        if x.shape[1] != self.mean.size:
            raise ValueError(f"each point must have dimension {self.mean.size}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")
        difference = x - self.mean
        mahalanobis_squared = np.einsum(
            "ni,ij,nj->n",
            difference,
            self.inv_cov,
            difference,
        )
        sign, log_determinant = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("cov determinant must be positive")
        dimension = self.mean.size
        log_normalization = -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant)
        log_density = log_normalization - 0.5 * mahalanobis_squared
        density = np.exp(log_density)
        return density.item() if single_point_sign else density

    def _draw(self, sample_size, rng):
        self._check_parameters()

        return rng.multivariate_normal(
            mean=self.mean,
            cov=self.cov,
            size=sample_size,
        )

    def _check_parameters(self):
        if self.mean is None:
            raise RuntimeError("mean is unknown")

        if self.cov is None:
            raise RuntimeError("covariance is unknown")
