import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e

from .rv import RandonVariable


class VonMises(RandonVariable):
    def __init__(self, mean_direction, concentration):
        super().__init__()
        self.mean_direction = mean_direction
        self.concentration = concentration

    @property
    def mean_direction(self):
        return self.parameters["mean_direction"]

    @mean_direction.setter
    def mean_direction(self, value):
        if not np.isscalar(value):
            raise TypeError("mean_direction must be a scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("mean_direction must be finite")

        self.parameters["mean_direction"] = value * 2 * np.pi

    @property
    def concentration(self):
        return self.parameters["concentration"]

    @concentration.setter
    def concentration(self, value):
        if not np.isscalar(value):
            raise TypeError("concentration must be a scalar")

        value = float(value)

        if not np.isfinite(value):
            raise ValueError("concentration must be finite")

        if value < 0:
            raise ValueError("concentration cannot be negative")

        self.parameters["concentration"] = value

    @property
    def mean_resultant_length(self):
        if self.concentration == 0:
            return 0.0

        return i1e(self.concentration) / i0e(self.concentration)

    @property
    def circular_variance(self):
        return 1.0 - self.mean_resultant_length

    def _pdf(self, x):
        difference = x - self.mean_direction

        # i0e(k) = exp(-abs(k)) * i0(k).
        # Because concentration >= 0:
        # log(i0(k)) = log(i0e(k)) + k.
        log_i0 = np.log(i0e(self.concentration)) + self.concentration

        log_normalization = np.log(2.0 * np.pi) + log_i0

        log_density = self.concentration * np.cos(difference) - log_normalization

        density = np.exp(log_density)

        return density.item() if density.ndim == 0 else density

    def _draw(self, sample_size, rng):
        return rng.vonmises(
            mu=self.mean_direction,
            kappa=self.concentration,
            size=sample_size,
        )

    def fit(self, observations):
        observations = self._validate_observations(observations)

        mean_cosine = np.mean(np.cos(observations))

        mean_sine = np.mean(np.sin(observations))

        self.mean_direction = np.arctan2(
            mean_sine,
            mean_cosine,
        )

        resultant_length = np.hypot(
            mean_cosine,
            mean_sine,
        )

        if resultant_length < 1e-12:
            self.concentration = 0.0
            return self

        if resultant_length >= 1.0 - 1e-12:
            raise ValueError(
                "concentration has no finite maximum-"
                "likelihood estimate because all angles "
                "are effectively identical"
            )

        self.concentration = self._estimate_concentration(resultant_length)

        return self

    @staticmethod
    def _estimate_concentration(
        resultant_length,
    ):
        def equation(concentration):
            if concentration == 0:
                ratio = 0.0
            else:
                ratio = i1e(concentration) / i0e(concentration)

            return ratio - resultant_length

        upper_bound = 1.0

        while equation(upper_bound) < 0:
            upper_bound *= 2.0

        return brentq(
            equation,
            a=0.0,
            b=upper_bound,
        )

    @staticmethod
    def _validate_observations(observations):
        try:
            observations = np.asarray(
                observations,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("observations must contain numeric angles") from exc

        if observations.ndim != 1:
            raise ValueError("observations must have shape (N,)")

        if observations.size == 0:
            raise ValueError("observations cannot be empty")

        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must contain only finite angles")

        return observations
