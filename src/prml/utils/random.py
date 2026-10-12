"""Random-number generator utilities."""

import numpy as np


def create_rng(seed=None):
    """Create a NumPy random generator.

    Parameters
    ----------
    seed : int, array_like, numpy.random.SeedSequence, or None, default=None
        Entropy used to initialize the generator.

    Returns
    -------
    numpy.random.Generator
        Initialized random generator.
    """
    return np.random.default_rng(seed)
