"""Basis-function design matrices for one-dimensional inputs."""

import numpy as np


def polynomial_basis(x: np.ndarray, degree: int):
    """Return polynomial basis values for one-dimensional inputs.

    Parameters
    ----------
    x : array_like of shape (n_samples,)
        Input values.
    degree : int
        Largest nonnegative polynomial degree.

    Returns
    -------
    numpy.ndarray of shape (n_samples, degree + 1)
        Design matrix containing powers from zero through ``degree``.
    """
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Degree must be non-negative integer.")
    return np.vander(x, degree + 1, increasing=True)


def gaussian_basis(x, centers, width):
    """Return Gaussian basis values for every input-center pair.

    Parameters
    ----------
    x : array_like of shape (n_samples,)
        Input values.
    centers : numpy.ndarray of shape (n_basis,)
        Centers of the Gaussian basis functions.
    width : float
        Positive common standard deviation.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_basis)
        Gaussian design matrix.
    """
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
    """Return logistic sigmoid basis values for each center.

    Parameters
    ----------
    x : array_like of shape (n_samples,)
        Input values.
    centers : numpy.ndarray of shape (n_basis,)
        Centers of the sigmoid basis functions.
    width : float
        Positive common scale.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_basis)
        Sigmoidal design matrix.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    if width <= 0:
        raise ValueError("width must be positive")
    values = (x[:, None] - centers[None, :]) / width
    return 1.0 / (1 + np.exp(-values))
