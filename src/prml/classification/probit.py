import numpy as np
from scipy.special import erf


class Probit:
    """Map a scalar or array of scores to class 1 probabilities."""

    def __call__(self, score):
        score = np.asarray(score, dtype=float)
        return 0.5 * (1.0 + erf(score / np.sqrt(2.0)))
