import numpy as np

from .rv import RandonVariable


class Gaussian(RandonVariable):
    def __init__(self, mean=None, standard_deviation=None):
        super().__init__()
        self.mean = mean
        self.standard_deviation = standard_deviation

    @property
    def mean(self):
        return self.parameters["mean"]

    @mean.setter
    def mean(self, value):
        if value == None:
            self.parameters["mean"] = None
            return
        if not np.isscalar(value):
            raise TypeError("value msut be a scalar")
        if not np.isfinite(value):
            raise ValueError("value cannot be infinite")
        self.parameters["mean"] = float(value)

    @property
    def standard_deviation(self):
        return self.parameters["standard_deviation"]

    @standard_deviation.setter
    def standard_deviation(self, value):
        if value == None:
            self.parameters["standard_deviation"] = None
            return
        if not np.isscalar(value):
            raise TypeError("value must be scalar")
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if value <= 0:
            raise ValueError("standard_deviation cannot be non-positive")
        self.parameters["standard_deviation"] = float(value)

    def _pdf(self, x):
        if self.mean == None or self.standard_deviation == None:
            raise RuntimeError("gaussian parameters are unknow; call fit first")
        normalization = 1 / (np.sqrt(2 * np.pi * self.standard_deviation**2))
        exponent = np.exp(-1 / (2 * self.standard_deviation**2) * (x - self.mean) ** 2)
        density = normalization * exponent
        return density

    def _draw(self, sample_size, rng):
        if self.mean == None or self.standard_deviation == None:
            raise RuntimeError("gaussian parameters are unknow; call fit first")
        return rng.normal(self.mean, self.standard_deviation, sample_size)

    def fit(self, observartions):
        pass
