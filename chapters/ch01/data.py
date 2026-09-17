import numpy as np


def sin_function(x):
    return np.sin(2 * np.pi * x)


def create_data(sample_size, noise_std, rng):
    x = np.linspace(0, 1, sample_size)
    noise = rng.normal(loc=0.0, scale=noise_std, size=sample_size)
    t = sin_function(x=x) + noise
    return x, t
