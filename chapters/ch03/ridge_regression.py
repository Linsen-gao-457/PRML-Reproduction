import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models.basis_functions import (
    gaussian_basis,
    polynomial_basis,
    sigmoidal_basis,
)
from prml.linear_models.ridge_regression import RidgeRegression
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(sample_size, std, domain: list[float], rng):
    x = np.linspace(domain[0], domain[1], sample_size)
    t = np.sin(2 * np.pi * x) + rng.normal(scale=std, size=sample_size)
    return x, t


def sinusoidal(x):
    return np.sin(2 * np.pi * x)


def create_figure(config, rng, x_train, t_train):
    x = np.linspace(
        config["plot"]["x_min"], config["plot"]["x_max"], config["plot"]["grid_points"]
    )
    true_value = sinusoidal(x)
    poly_matrix_train = polynomial_basis(
        x=x_train, degree=config["polynomial"]["degree"]
    )
    gaussian_centers = np.linspace(
        config["plot"]["x_min"],
        config["plot"]["x_max"],
        config["gaussian"]["number_of_centers"],
    )
    gaussian_matrix_train = gaussian_basis(
        x=x_train, width=config["gaussian"]["width"], centers=gaussian_centers
    )
    sigmoid_centers = np.linspace(
        config["plot"]["x_min"],
        config["plot"]["x_max"],
        config["sigmoid"]["number_of_centers"],
    )
    sigmoid_matrix_train = sigmoidal_basis(
        x=x_train, centers=sigmoid_centers, width=config["sigmoid"]["width"]
    )
    training_matrices = [poly_matrix_train, gaussian_matrix_train, sigmoid_matrix_train]
    plot_matrices = [
        polynomial_basis(x=x, degree=config["polynomial"]["degree"]),
        gaussian_basis(x, gaussian_centers, width=config["gaussian"]["width"]),
        sigmoidal_basis(x, sigmoid_centers, config["sigmoid"]["width"]),
    ]
    figure, axes = plt.subplots(
        3,
        1,
        sharex=True,
        sharey=True,
    )
    titles = [
        "Polynomial basis",
        "Gaussian basis",
        "Sigmoidal basis",
    ]

    for axis, training_matrix, plot_matrix, title in zip(
        axes.flat, training_matrices, plot_matrices, titles
    ):
        model = RidgeRegression(regularization=config["regularization"])
        model.fit(training_matrix, t_train)
        prediction = model.predict(plot_matrix)
        axis.plot(
            x,
            true_value,
            color="green",
            label="Ground truth",
        )
        axis.plot(x, prediction, color="red", label="Ridge regression")
        axis.scatter(
            x_train,
            t_train,
            facecolor="none",
            edgecolor="blue",
            label="Training data",
        )
        axis.set_title(title)
        axis.set_xlim(
            config["plot"]["x_min"],
            config["plot"]["x_max"],
        )
        axis.tick_params(
            direction="in",
            top=True,
            right=True,
        )
    axes[0].legend()
    axes[-1].set_xlabel("$x$")
    figure.tight_layout()

    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/ridge_regression.yaml")

    rng = create_rng(seed=config["seed"])

    x_train, t_train = create_data(
        sample_size=config["data"]["sample_size"],
        std=config["data"]["std"],
        rng=rng,
        domain=[config["plot"]["x_min"], config["plot"]["x_max"]],
    )
    figure = create_figure(config=config, rng=rng, x_train=x_train, t_train=t_train)

    output = save_figure(
        figure,
        PROJECT_ROOT / config["output"],
    )

    plt.close(figure)
    return output


def main():
    run()


if __name__ == "__main__":
    main()
