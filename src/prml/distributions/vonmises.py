"""Von Mises distribution for periodic angular observations."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e

from .rv import RandonVariable


class VonMises(RandonVariable):
    """Represent, sample, and fit a circular distribution.

    Parameters
    ----------
    mean_direction : float
        Finite mean direction.
    concentration : float
        Nonnegative concentration parameter.

    Attributes
    ----------
    mean_resultant_length : float
        Population mean resultant length.
    circular_variance : float
        One minus the mean resultant length.
    """

    def __init__(self, mean_direction, concentration):
        """Initialize mean direction and concentration."""
        super().__init__()
        self.mean_direction = mean_direction
        self.concentration = concentration

    @property
    def mean_direction(self):
        """Return the mean direction."""
        return self.parameters["mean_direction"]

    @mean_direction.setter
    def mean_direction(self, value):
        """Set the finite mean direction."""
        if not np.isscalar(value):
            raise TypeError("mean_direction must be a scalar")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("mean_direction must be finite")

        self.parameters["mean_direction"] = value * 2 * np.pi

    @property
    def concentration(self):
        """Return the concentration parameter."""
        return self.parameters["concentration"]

    @concentration.setter
    def concentration(self, value):
        """Set the nonnegative concentration parameter."""
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
        """Return the population mean resultant length."""
        if self.concentration == 0:
            return 0.0

        return i1e(self.concentration) / i0e(self.concentration)

    @property
    def circular_variance(self):
        """Return the circular variance."""
        return 1.0 - self.mean_resultant_length

    def _pdf(self, x):
        """Evaluate the von Mises density.

        Parameters
        ----------
        x : numpy.ndarray
            Angular evaluation points.

        Returns
        -------
        float or numpy.ndarray
            Density at each angle.
        """
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
        """Draw angular samples."""
        return rng.vonmises(
            mu=self.mean_direction,
            kappa=self.concentration,
            size=sample_size,
        )

    def fit(self, observations):
        """Estimate parameters by maximum likelihood.

        Parameters
        ----------
        observations : array_like of shape (n_samples,)
            Nonempty finite angular observations.

        Returns
        -------
        VonMises
            Fitted distribution.

        Raises
        ------
        ValueError
            If identical angles imply no finite concentration estimate.
        """
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
        """Invert the Bessel-function ratio.

        Parameters
        ----------
        resultant_length : float
            Sample mean resultant length strictly between zero and one.

        Returns
        -------
        float
            Maximum-likelihood concentration estimate.
        """
        def equation(concentration):
            """Evaluate the scalar concentration root equation.

            Parameters
            ----------
            concentration : float
                Candidate nonnegative concentration.

            Returns
            -------
            float
                Difference between the implied and observed resultant lengths.
            """
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
        """Validate angular observations.

        Parameters
        ----------
        observations : array_like
            Candidate angular observations.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Finite floating-point angles.
        """
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
