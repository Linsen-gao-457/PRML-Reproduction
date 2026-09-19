"""Placeholder entry point for the polynomial RMS error comparison."""

import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models import LinearRegression, polynomial_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng

from .figure_1_2 import generate_data


def create_figure(
    train_errors: np.ndarray, test_errors: np.ndarray, degrees: np.ndarray
):
    figure, axis = plt.subplots()
    axis.plot(
        degrees,
        train_errors,
        "-o",
        color="blue",
        mfc="white",
        mec="blue",
        label="Train",
    )
    axis.plot(
        degrees,
        test_errors,
        "-o",
        color="red",
        mfc="white",
        mec="red",
        label="Test",
    )
    axis.set(xlabel="$M$", ylabel="E_{RMS}", xlim=[-0.5, 10], ylim=[-0.1, 1])
    axis.set_xticks = [0, 3, 6, 9]
    axis.legend()
    return figure


def rms(predictions, targets):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_errors(
    max_degree: int,
    x_train: np.ndarray,
    t_train: np.ndarray,
    x_test: np.ndarray,
    t_test: np.ndarray,
):
    train_errors = []
    test_errors = []
    for degree in range(max_degree + 1):
        design_matrix_train = polynomial_basis(x_train, degree)
        design_matrix_test = polynomial_basis(x_test, degree)
        model = LinearRegression()
        model.fit(design_matrix_train, t_train)
        training_prediction = model.predict(design_matrix_train)
        test_prediction = model.predict(design_matrix_test)
        train_error = rms(predictions=training_prediction, targets=t_train)
        testing_error = rms(predictions=test_prediction, targets=t_test)
        train_errors.append(train_error)
        test_errors.append(testing_error)
    return np.asarray(train_errors), np.asarray(test_errors)


def run():
    configure_path = CONFIGS_DIR / "ch01/figure_1_5.yaml"
    configure = load_config(configure_path)
    rng = create_rng(seed=configure["seed"])
    train_size, test_size = configure["sample_sizes"]
    x_train, t_train = generate_data(
        sample_size=train_size, noise_std=configure["noise_std"], rng=rng
    )
    x_test, t_test = generate_data(
        sample_size=test_size, noise_std=configure["noise_std"], rng=rng
    )
    degrees = np.arange(start=0, stop=configure["max_degree"] + 1)
    train_errors, test_errors = calculate_errors(
        configure["max_degree"], x_train, t_train, x_test, t_test
    )
    figure = create_figure(train_errors, test_errors, degrees)
    output = save_figure(figure, PROJECT_ROOT / configure["output"])
    plt.close(figure)
    print(f"Saved: {output}")
    return output


def main():
    run()


if __name__ == "__main__":
    main()
