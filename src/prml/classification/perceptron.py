import numpy as np


class PerceptronClassifier:
    """Binary classifier using the perceptron learning algorithm.

    Parameters
    ----------
    learning_rate : float, default=1.0
        Positive multiplier applied to every weight update.
    max_epochs : int, default=1000
        Maximum number of passes through the training data.
    fit_intercept : bool, default=False
        Whether to include an intercept parameter.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (2,) or None
        Class labels observed during fitting.
    coef_ : numpy.ndarray of shape (n_features,) or None
        Fitted feature weights.
    intercept_ : float or None
        Fitted intercept.
    history_ : list of dict
        Weight-update records.
    converged_ : bool
        Whether an epoch completed without a mistake.
    """

    def __init__(self, learning_rate=1.0, max_epochs=1000, fit_intercept=False):
        if not np.isscalar(learning_rate):
            raise TypeError("learning_rate must be a scalar")

        learning_rate = float(learning_rate)

        if not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.fit_intercept = bool(fit_intercept)

        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.history_ = None
        self.converged_ = None
        self.n_epochs_ = None

        self._weights = None
        self._n_features = None

    def fit(self, featrues, targets, initial_weight=None):
        """Fit the perceptron.

        Parameters
        ----------
        features : array_like of shape (n_samples, n_features)
            Training observations.
        targets : array_like of shape (n_samples,)
            Binary class labels.
        initial_weights : array_like or None, default=None
            Initial model weights. The expected length is
            ``n_features + 1`` when fitting an intercept and
            ``n_features`` otherwise. If ``None``, zero initialization
            is used.

        Returns
        -------
        PerceptronClassifier
            Fitted classifier.
        """
        featrues, targets = self._validate_training_data(featrues, targets)
        self.classes_, target_indices = np.unique(targets, return_inverse=True)
        if self.classes_.size != 2:
            raise ValueError("perceptron classification requires exactly two classes")
        signed_targets = np.where(
            target_indices == 0,
            -1,
            1,
        )
        design_matrix = self._create_design_matrix(featrues)
        if initial_weight is None:
            weights = np.zeros(design_matrix.shape[1], dtype=float)
        else:
            weights = np.asarray(initial_weight, dtype=float).copy()

        if weights.shape != (design_matrix.shape[1],):
            raise ValueError("initial_weights has the wrong shape")

        if not np.all(np.isfinite(weights)):
            raise ValueError("initial_weights must contain only finite values")
        self.history_ = []
        self.converged_ = False
        for epoch in range(self.max_epochs):
            number_of_updates = 0
            for index, (design_row, target) in enumerate(
                zip(design_matrix, signed_targets)
            ):
                score = np.dot(weights, design_row)
                if score * target > 0:
                    continue
                weights_before = weights.copy()
                update = self.learning_rate * target * design_row
                weights += update
                number_of_updates += 1
                self.history_.append(
                    {
                        "epoch": epoch,
                        "index": index,
                        "class_label": target,
                        "feature": featrues[index].copy(),
                        "weights_before": weights_before,
                        "update": update.copy(),
                        "weights_after": weights.copy(),
                    }
                )
            self.n_epochs_ = epoch + 1
            if number_of_updates == 0:
                self.converged_ = True
                break
        self._weights = weights
        self._n_features = featrues.shape[1]
        if self.fit_intercept:
            self.intercept_ = float(weights[0].copy())
            self.coef_ = weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.copy()
        return self

    def _create_design_matrix(self, features):
        if not self.fit_intercept:
            return features

        return np.column_stack(np.ones(features.shape[0]), features)

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
                "features and targets must contain the same number of samples"
            )

        if features.shape[0] == 0:
            raise ValueError("training data must not be empty")

        if features.shape[1] == 0:
            raise ValueError("features must contain at least one feature")

        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")

        return features, targets
