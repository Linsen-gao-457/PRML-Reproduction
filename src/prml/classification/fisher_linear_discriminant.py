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

    def fit(self, features, targets):
        """Estimate the Fisher projection direction
        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Training observations.
        targets : array_like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        FisherLinearDiscriminant
            Fitted discriminant.
        """
        features, targets = self._validate_training_data(
            features,
            targets,
        )
        self.classes_ = np.unique(targets)

        if self.classes_.size != 2:
            raise ValueError("FisherLinearDiscriminant requires exactly two classes")
        class_features = [
            features[targets == class_label] for class_label in self.classes_
        ]
        self.class_means = np.vstack(
            [observations.mean(axis=0) for observations in class_features]
        )
        number_of_features = features.shape[1]
        within_class_scatter = np.zeros(
            (number_of_features, number_of_features), dtype=float
        )
        for observations, class_mean in zip(class_features, self.class_means):
            centered = observations - class_mean
            within_class_scatter = centered.T @ centered
        self.within_class_scatter_ = within_class_scatter
        regularized_scatter = within_class_scatter + self.regularization * np.eye(
            number_of_features
        )
        mean_difference = self.class_means[1] - self.class_means[0]
        direction = np.linalg.solve(regularized_scatter, mean_difference)
        direction_norm = np.linalg.norm(direction)
        if np.isclose(direction_norm, 0.0):
            raise ValueError("the fitted Fisher direction has zero length")
        self.direction_ = direction / direction_norm
        self._n_features = number_of_features

        return self

    def transform(self, features):
        """Project observations onto the Fisher direction.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Observations to project.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            One-dimensional projected values.
        """
        features = self._validate_prediction_data(features)
        return features @ self.direction_

    def fit_transform(self, features, targets):
        """Fit the discriminant and project the training data."""
        return self.fit(features, targets).transform(features)

    def fisher_criterion(self):
        """Return the Fisher separation criterion for the fitted direction."""
        if self.direction_ is None:
            raise RuntimeError("fit must be called before fisher_criterion")

        projected_means = self.class_means_ @ self.direction_
        numerator = (projected_means[1] - projected_means[0]) ** 2

        denominator = self.direction_ @ self.within_class_scatter_ @ self.direction_

        if np.isclose(denominator, 0.0):
            return np.inf

        return numerator / denominator

    @staticmethod
    def _validate_training_data(features, targets):
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets)

        if features.ndim != 2:
            raise ValueError("features must be two-dimensional")

        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")

        if features.shape[0] != targets.size:
            raise ValueError(
                "features and targets must have the same number of samples"
            )

        if features.shape[0] == 0:
            raise ValueError("training data must not be empty")

        if features.shape[1] == 0:
            raise ValueError("features must contain at least one feature")

        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")

        return features, targets

    def _validate_prediction_data(self, features):
        if self.direction_ is None:
            raise RuntimeError("fit must be called before transform")

        features = np.asarray(features, dtype=float)

        if features.ndim != 2:
            raise ValueError("features must be two-dimensional")

        if features.shape[1] != self._n_features:
            raise ValueError(f"features must contain {self._n_features} columns")

        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")

        return features
