import numpy as np
from scipy.special import gamma

from .rv import RandomVariable


class NormalGamma(RandomVariable):
    def __init__(self, beta, mu_0, a, b):
        super().__init__()
        self.mu_0 = mu_0
        self.beta = beta
        self.a = a
        self.b = b

    @classmethod
    def form_constraints(cls, c, beta, d):
        mu_0 = c / beta
        a = 1 + beta / 2
        b = d - c**2 / (2 * beta)
        return cls(beta=beta, mu_0=mu_0, a=a, b=b)

    @property
    def mu_0(self):
        return self.parameters["mu_0"]

    @mu_0.setter
    def mu_0(self, value):
        if not np.isscalar(value):
            raise TypeError("mu_0 must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("mu_0 must be finite")

        self.parameters["mu_0"] = value

    @property
    def beta(self):
        return self.parameters["beta"]

    @beta.setter
    def beta(self, value):
        if not np.isscalar(value):
            raise TypeError("beta must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("beta must be finite")

        if value <= 0:
            raise ValueError("beta must be positive")

        self.parameters["beta"] = value

    @property
    def a(self):
        return self.parameters["a"]

    @a.setter
    def a(self, value):
        if not np.isscalar(value):
            raise TypeError("a must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("a must be finite")

        if value <= 0:
            raise ValueError("a must be positive")
        expected_a = 1.0 + self.beta / 2.0

        if not np.isclose(value, expected_a):
            raise ValueError(f"constraint requires a = 1 + beta / 2 = {expected_a}")

        self.parameters["a"] = value

    @property
    def b(self):
        return self.parameters["b"]

    @b.setter
    def b(self, value):
        if not np.isscalar(value):
            raise TypeError("b must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("b must be finite")

        if value <= 0:
            raise ValueError("b must be positive")

        self.parameters["b"] = value

    def _pdf(self, x):
        x = np.asarray(x, dtype=float)
        single_point_sign = x.ndim == 1
        if single_point_sign and x.shape != (2,):
            raise ValueError("single point must have shape (2, )")
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("value must have shape (N,2)")

        mu = x[:, 0]
        precision = x[:, 1]
        if np.any(precision) < 0:
            raise ValueError("precision must be positive")
        gamma_density = self.b**self.a / gamma(self.a) * precision ** (self.a - 1) np.exp(-self.b * precision)
        gaussian_density = np.sqrt(self.beta * precision/ (2 * np.pi)) * np.exp(-0.5*self.beta*precision *(mu - self.mu_0)**2)
        density = gamma_density * gaussian_density
        return density[0] if single_point_sign else density

    def _draw(self, sample_size, rng=None):
        pass



# def draw(self, sample_size, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     precision = rng.gamma(
#         shape=self.a,
#         scale=1.0 / self.b,
#         size=sample_size,
#     )

#     standard_deviation = np.sqrt(1.0 / (self.beta * precision))

#     mean = rng.normal(
#         loc=self.m,
#         scale=standard_deviation,
#         size=sample_size,
#     )

#     return np.column_stack((mean, precision))
