"""Categorical distribution for one-hot encoded observations."""

import numpy as np

from .dirichlet import Dirichlet
from .rv import RandonVariable


class Categorical(RandonVariable):
    """Model categorical observations.

    Parameters
    ----------
    mu : array_like, Dirichlet, or None, default=None
        Probability vector, conjugate Dirichlet prior, or ``None`` when the
        probabilities will be estimated later.

    Attributes
    ----------
    probabilities : numpy.ndarray of shape (n_classes,)
        Point probabilities or mean of the Dirichlet prior/posterior.
    size : int or None
        Number of categories when known.
    """

    def __init__(self, mu=None):
        """Initialize probabilities or a conjugate prior."""
        super().__init__()
        self.k_class = None
        self.mu = mu

    @property
    def mu(self):
        """Return the probability vector or Dirichlet prior."""
        return self.parameters["mu"]

    @mu.setter
    def mu(self, mu):
        """Set probabilities or a Dirichlet prior."""
        if mu is None:
            self.parameters["mu"] = None
            return
        if isinstance(mu, Dirichlet):
            self.k_class = mu.size
            self.parameters["mu"] = mu
            return
        mu = np.asarray(mu, "float")
        if isinstance(mu, np.ndarray):
            if mu.ndim != 1:
                raise ValueError("dismensionallity of mu must be 1")
            if (mu < 0).any():
                raise ValueError()
            if not np.allclose(mu.sum(), 1):
                raise ValueError("sum of mu must be 1")
            self.k_class = mu.size
            self.parameters["mu"] = mu.copy()

    @property
    def probabilities(self):
        """Return point probabilities or the prior/posterior mean."""
        if self.mu is None:
            raise RuntimeError("mu is unknown")
        if isinstance(self.mu, Dirichlet):
            return self.mu.mean
        return self.mu

    @property
    def size(self):
        """Return the number of categories when known."""
        if hasattr(self.mu, "size"):
            return self.k_class

    @property
    def mean(self):
        """Return the mean one-hot vector."""
        return self.probabilities

    def pmf(self, x):
        """Evaluate masses for one-hot observations.

        Parameters
        ----------
        x : array_like of shape (n_classes,) or (n_samples, n_classes)
            One-hot encoded observations.

        Returns
        -------
        float or numpy.ndarray of shape (n_samples,)
            Probability mass for each observation.
        """
        observations, single_observation_sign = self._validate_observations(
            observations=x
        )
        pmf = observations @ self.probabilities
        if single_observation_sign:
            return pmf[0]
        return pmf

    def fit(self, observations):
        """Fit probabilities or update a Dirichlet prior.

        Parameters
        ----------
        observations : array_like of shape (n_samples, n_classes)
            One-hot encoded observations.

        Returns
        -------
        Categorical
            Fitted distribution.
        """
        observations, _ = self._validate_observations(observations)
        num_class = observations.sum(axis=0)
        if isinstance(self.mu, np.ndarray):
            self.mu = num_class / num_class.sum()
        elif isinstance(self.mu, Dirichlet):
            self.mu = Dirichlet(self.mu.alpha + num_class)
        return self

    def _pdf(self, x):
        """Evaluate mass through the common density interface."""
        return self.pmf(x)

    def _draw(self, sample_size, rng):
        """Draw one-hot encoded categorical samples."""
        class_indices = rng.choice(
            self.k_class,
            size=sample_size,
            p=self.probabilities,
        )

        return np.eye(
            self.k_class,
            dtype=int,
        )[class_indices]

    def _validate_observations(
        self,
        observations,
    ):
        """Validate one-hot encoded observations.

        Parameters
        ----------
        observations : array_like
            Candidate observation or observation matrix.

        Returns
        -------
        observations : numpy.ndarray of shape (n_samples, n_classes)
            Validated integer observations.
        single_observation : bool
            Whether the original input represented one observation.
        """
        observations = np.asarray(observations)
        single_obervation_sign = observations.ndim == 1
        if single_obervation_sign:
            observations = observations.reshape(1, -1)
        if observations.ndim != 2:
            raise ValueError("observations must have shape (n_samples, k_classes)")
        if observations.shape[0] == 0:
            raise ValueError("Observation canot be empty.")
        if observations.shape[1] < 2:
            raise ValueError("Observations should contain at least two classes.")
        if self.k_class is not None and observations.shape[1] != self.k_class:
            raise ValueError(
                f"expected {self.k_class} classes, received {observations.shape[1]}"
            )
        if not np.all((observations == 0) | (observations == 1)):
            raise ValueError("each element of observations should be 0 or 1")
        if not np.all(observations.sum(axis=1) == 1):
            raise ValueError("each obervation must contain one active class")
        return (observations.astype(int), single_obervation_sign)
