import numpy as np

from .rv import RandonVariable


class Dirichlet(RandonVariable):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    @property
    def alpha(self):
        return self.parameters["alpha"]

    @alpha.setter
    def alpha(self, value):
        value = np.asarray(value, dtype=float)
        if value.ndim != 1:
            raise ValueError("alpha must contain at least two values")
        if value.size < 2:
            raise ValueError("alpha must contain at least 2 values")
        if not np.any(value <= 0):
            raise ValueError("alpha must be positive")
        if not np.all(np.isfinite(value)):
            raise ValueError("alpha cannot contain finite values")
        self.parameters["alpha"] = value.copy()
