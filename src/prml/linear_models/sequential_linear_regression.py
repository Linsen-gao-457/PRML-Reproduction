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

    def partial_fit(self, designed_matrix, targets):
        pass

    def fit(self):
        pass

    def predict(self):
        pass
