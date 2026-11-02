import numpy as np


class FisherLinearDiscriminant:
    """a discriminant method to find a one-dimensional projection for two-class separation
    Parameters
    ----------
    regularization : float, default=1e-8
        Nonnegative diagonal regularization added to the within-class
        scatter matrix.

    Attributes
    ----------
    classes_: numpy.ndarray of shape (2, ) or None
    class_means: numpy.ndarray of shape (2, n_features) or None.
    Mean feature vector for each class.
    direction_: numpy.ndarray of shape (n_features,) or None
        Unit-length Fisher projection direction.
    """

    def __init__(self, regularization=1.0e-8):
        if not np.isscalar(regularization):
            raise TypeError("regularization must be a scalar")
        regularization = float(regularization)
        if not np.isfinite(regularization):
            raise ValueError("regularization must be finite")

        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        self.regularization = regularization
        self.classes = None
        self.class_means = None
        self.direction_ = None
        self._n_features = None
        self.within_class_scatter_ = None

    def fit(features, targets):
        pass
