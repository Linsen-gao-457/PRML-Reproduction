import numpy as np

from .rv import RandonVariable


class MultivariateGaussian(RandonVariable):
    def __init__(self):
        super().__init__()

    def _pdf(self, x):
        return super()._pdf(x)

    def _draw(self, sample_size, rng):
        return super()._draw(sample_size, rng)
