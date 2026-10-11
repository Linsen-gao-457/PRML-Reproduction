import numpy as np


class SequentialLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.coef_ = None

    def partial_fit(self, phi_x, t):
        phi_x = np.asarray(phi_x, dtype=float)
        t = np.asarray(t, dtype=float)
        prediction = np.dot(phi_x, self.coef_)
        error = t - prediction
        self.coef_ += self.learning_rate * error * phi_x

    def fit(self, designed_matrix, targets):
        designed_matrix = np.asanyarray(designed_matrix, dtype=float)
        targets = np.asanyarray(targets, dtype=float)
        if designed_matrix.ndim != 2:
            raise ValueError("design_matrix must be two-dimensional")

        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")

        if designed_matrix.shape[0] != targets.size:
            raise ValueError(
                "design_matrix and targets must have the same number of observations"
            )
        if not np.all(np.isfinite(designed_matrix)):
            raise ValueError("design_matrix must contain finite values")

        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must contain finite values")
        self.coef_ = np.zeros(designed_matrix.shape[1], dtype=float)
        for _ in range(self.epochs):
            for phi_x, t in zip(designed_matrix, targets):
                self.partial_fit(phi_x, t)
        return self

    def predict(self):
        pass
