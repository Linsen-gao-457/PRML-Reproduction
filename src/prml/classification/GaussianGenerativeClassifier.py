import numpy as np


class GaussianGenerativeClassifier:
    """
    Model each conditional class P(x|C) with a multivariate Gaussian distribution and derive posterior distribution
    Parameters
    ------
    reg : float, default=1e-6
        Nonnegative value added to the covariance diagonal for numerical
        stability.
    ------
    m_features_: int
    k_classes_: int
    classes_ : ndarray of shape (k_classes, )
            Class labels in probability-column order
    priors_ : ndarray of shape (k_classes, )
        Estimate prior probability of each class
    means_: ndarray of shape (k_classes, m_features)
        Estimate mean vector of each class
    covariance_: ndarray of shape (m_features, m_features)
    coef_: ndaaray of shape (k_classes, m_features)
    intercept_: ndaaray of shape (k_classes)
    """

    def __init__(self, reg=1e-6):
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and nonnegative")
        self.reg = reg
        self.k_classes_ = None
        self.m_features_ = None
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, Y):
        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass

    def sample(self, n_samples, random_state=None):
        pass
