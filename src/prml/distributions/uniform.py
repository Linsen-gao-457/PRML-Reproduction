import numpy as np

from .rv import RandonVariable


class Uniform(RandonVariable):
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    @property
    def low(self):
        return self.parameters["low"]

    @low.setter
    def low(self, value):
        if not np.isscalar(value):
            raise TypeError("low must be scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("low cannot be infinite")
        if "high" in self.parameters and value >= self.high:
            raise ValueError("low must be smaller than high")
        self.parameters["low"] = value

    @property
    def high(self):
        return self.parameters["high"]

    @high.setter
    def high(self, value):
        if not np.isscalar(value):
            raise TypeError("high must be a scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("high must be finite")
        if "low" in self.parameters and value <= self.low:
            raise ValueError("high must be larger than low")
        self.parameters["high"] = value

    def _pdf(self, x):
        density = 1.0 / (self.high - self.low)
        result = np.where((x >= self.low) & (x <= self.high), density, 0.0)
        return result.item() if result.ndim == 0 else result

    def _draw(self, sample_size, rng):
        return rng.uniform(low=self.low, high=self.high, size=sample_size)
