import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


class LaplaceApproximation:
    """Build a local Gaussian approximation to a continuous density."""

    def __call__(self, f, initial):
        """Approximate the density proportional to f(z).

        Parameters
        ----------
        f : callable
            Unnormalized target density. Takes a vector of shape (m,)
            and must return a finite, positive scalar at evaluated points.
        initial : array-like of shape (m,)
            Starting point for finding a mode.

        Returns
        -------
        gaussian : scipy.stats distribution
            Frozen multivariate Gaussian with ``mean``, ``cov``,
            ``pdf``, ``logpdf``, and ``rvs``.
        """
        initial = np.asarray(initial, dtype=float)
        if initial.ndim != 1 or initial.size == 0:
            raise ValueError("initial must be a nonempty vector")
        if not np.isfinite(initial).all():
            raise ValueError("initial must contain only finite values")

        def objective(z):
            value = f(z)
            if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
                raise ValueError("f(z) must return a finite, positive scalar")
            return -np.log(value)

        result = minimize(objective, initial, method="BFGS", jac="3-point")
        if not result.success:
            raise RuntimeError(f"Could not find a mode: {result.message}")
        mode = result.x
        m = mode.size
        f0 = objective(mode)

        # Central finite differences estimate the Hessian at the mode.
        step = np.finfo(float).eps ** 0.25 * np.maximum(1.0, np.abs(mode))
        offsets = np.diag(step)
        hessian = np.empty((m, m))

        for i, di in enumerate(offsets):
            hessian[i, i] = (
                objective(mode + di) - 2 * f0 + objective(mode - di)
            ) / step[i] ** 2

            for j in range(i):
                dj = offsets[j]
                mixed = (
                    objective(mode + di + dj)
                    - objective(mode + di - dj)
                    - objective(mode - di + dj)
                    + objective(mode - di - dj)
                ) / (4 * step[i] * step[j])
                hessian[i, j] = hessian[j, i] = mixed

        try:
            factor = cho_factor(hessian, lower=True)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The estimated Hessian must be positive definite."
            ) from exc

        covariance = cho_solve(factor, np.eye(m))
        return multivariate_normal(mean=mode, cov=covariance)
