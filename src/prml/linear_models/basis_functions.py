import numpy as np


def polynomial_basis(x: np.ndarray, degree: int):
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Degree must be non-negative integer.")
    return np.vander(x, degree + 1, increasing=True)


def gaussian_basis(x, centers, width):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    if width <= 0:
        raise ValueError("width must be positive")
    difference = x[:, None] - centers[None, :]
    return np.exp(-0.5 * (difference / width) ** 2)


def sigmoidal_basis(x, centers, width):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    if width <= 0:
        raise ValueError("width must be positive")
    values = (x[:, None] - centers[None, :]) / width
    return 1.0 / (1 + np.exp(values))
