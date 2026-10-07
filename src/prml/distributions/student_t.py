import numpy as np

from .rv import RandonVariable

import scipy.special import digamma, gammaln


class Student_T(RandonVariable):
    def __init__(self, mean =0.0, precision =1.0, degree_of_freedom =1.0):
        super().__init__()
        self.mean = mean
        self.precision = precision
        self.degree_of_freedom = degree_of_freedom
    @property
    def mean(self):
        return self.parameters['mean']
    
    @mean.setter
    def mean(self, value):
        if not np.isscalar(value):
            raise TypeError("mean must be a scalar")
        
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("mean must be finite")
        
        self.parameters["mean"] = value
    
    @property
    def precision(self):
        return self.parameters["precision"]
    
    def precision(self, value):
        if not np.isscalar(value):
            raise TypeError(
                "precision must be a scalar"
            )

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "precision must be finite"
            )

        if value <= 0:
            raise ValueError(
                "precision must be positive"
            )

        self.parameters["precision"] = value
    
    @property
    def degree_of_freedom(self):
        return self.parameters["degree_of_freedom"]

    @degree_of_freedom.setter
    def degree_of_freedom(self, value):
        if not np.isscalar(value):
            raise TypeError(
                "degree_of_freedom must be a scalar"
            )

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "degree_of_freedom must be finite"
            )

        if value <= 0:
            raise ValueError(
                "degree_of_freedom must be positive"
            )

        self.parameters["degree_of_freedom"] = value

    def _pdf(self, x):
        nu = self.degree_of_freedom
        squared_error = (x-self.mean)**2
        log_normalization = (gammaln((nu + 1)/2 -gammaln(nu/2.0)) + 0.5* (np.log(self.precision) - np.log(np.pi * nu)))
        log_kernel = -(nu + 1)/2 * np.log1p(self.precision *squared_error /nu)
        density = np.exp(log_kernel + log_normalization)
        return density.item() if density.ndim ==1 else density
        
    def _draw(self, sample_size, rng):
        scale = 1.0 / np.sqrt(self.precision)
        return self.mean + scale * rng.standard_t(
            df=self.degree_of_freedom,
            size=sample_size,
        )
        
    def fit():
        pass
