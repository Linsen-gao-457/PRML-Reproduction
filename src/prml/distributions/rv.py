from abc import ABC, abstractmethod

import numpy as np


class RandonVariable(ABC):
    def __init__(self):
        self.parameters = {}

    def __repr__(self):
        arguments = ", ".join(
            f"{name}={value!r}" for name, value in self.parameters.items()
        )
        return f"{self.__class__.__name__}({arguments})"

    def pdf(self, x):
        x = np.asarray(x, dtype="float")
        return self._pdf(x)

    def draw(self, sample_size=1, rng=None):
        if not isinstance(sample_size, (int, np.integer)):
            raise TypeError("sample_size must be an integer")
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        rng = rng or np.random.default_rng()
        return self._draw(sample_size, rng)

    @abstractmethod
    def _pdf(self, x):
        pass

    @abstractmethod
    def _draw(self, sample_size, rng):
        pass
