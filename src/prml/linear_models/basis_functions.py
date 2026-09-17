import numpy as np


def polynomial_basis(x: np.ndarray, degree: int):
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Degree must be non-negative integer.")
    return np.vander(x, degree + 1, increasing=True)
