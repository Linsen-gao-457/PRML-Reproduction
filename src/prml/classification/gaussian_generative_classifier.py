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
        """Estimate parameters mean, covatiance, priors

        Args:
        -------
        X : array-like of shape (n_samples, m_features)
            Finite training features.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : GaussianGenerativeClassifier
            Fitted classifier.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        n_samples, m_features = X.shape
        if n_samples == 0 or m_features == 0:
            raise ValueError("X must contain samples and features.")
        if Y.shape != (n_samples,):
            raise ValueError("Y must have shape (n_samples,).")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")
        classes, index = np.unique(Y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Training data must contain at least two classes.")

        self.classes_ = classes
        self.k_classes_ = len(classes)
        self.m_features_ = m_features
        counts = np.bincount(index, minlength=self.k_classes_)
        self.priors_ = counts / n_samples
        self.means_ = np.stack(
            ([X[index == k].mean(axis=0) for k in range(self.k_classes_)]),
            axis=0,
            dtype=float,
        )
        centered = X - self.means_[index]
        self.covariance_ = centered.T @ centered / n_samples
        self.covariance_ += self.reg * np.eye(self.m_features_)
        self.coef_ = np.linalg.solve(self.covariance_, self.means_.T).T
        self.intercept_ = np.asarray(
            [
                -0.5 * self.means_[k] @ self.coef_ + np.log(self.priors_[k])
                for k in range(self.k_classes_)
            ],
            dtype=float,
        )
        return self

    def predict_proba(self, X):
        """compute posterior class probability

        Args:
            X (ndarray): shape (n_samples, k_classes)
            posterior probability in "classes" order

        Returns
        -------
        probabilities : ndarray of shape (n_samples, k_classes)
            Posterior probabilities in ``classes_`` order.
        """
        if not hasattr(self, "_coef"):
            raise RuntimeError("Call fit before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.m_features_:
            raise ValueError(
                f"X must be two-dimensional with {self.m_features_} features."
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")
        scores = X @ self.coef_.T + self.intercept_
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict the most probable class.

        Parameters
        ----------
        X : array-like of shape (n_samples, m_features)
            Input features.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted original class labels.
        """
        prob = self.predict(X)
        indices = np.argmax(prob, axis=1)
        return self.classes_[indices]

    def sample(self, n_samples=100, random_state=None):
        """Generate a synthetic dataset from the fitted model.
         Parameters
         ----------
         n_samples : int, default=100
             Positive number of samples to generate.
         random_state : int or None, default=None
             Seed for reproducible random sampling.
         Returns
         -------
         X : ndarray of shape (n_samples, m_features)
             Generated feature vectors.
        Y : ndarray of shape (n_samples,)
             Generated labels using the original class labels.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before sampling.")

        if (
            isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(random_state)
        class_indices = rng.choice(self.k_classes_, size=n_samples, p=self.priors_)
        signal = self.means_[class_indices]
        noise = rng.multivariate_normal(
            np.zeros(self.m_features_),
            cov=self.covariance_,
            size=n_samples,
        )
        X = signal + noise
        Y = self.classes_[class_indices]
        return X, Y
