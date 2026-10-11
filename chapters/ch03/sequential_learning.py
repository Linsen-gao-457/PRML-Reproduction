import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models.basis_functions import (
    gaussian_basis,
    polynomial_basis,
    sigmoidal_basis,
)
from prml.linear_models.sequential_linear_regression import SequentialLinearRegression
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


def create_figure(config, x_train, t_train):
    x = np.linspace(
        config["plot"]["x_min"], config["plot"]["x_max"], config["plot"]["grid_points"]
    )
    true_values = sinusoidal(x)
    polynomial_degree = config["polynomial"]["degree"]

    gaussian_centers = np.linspace(
        config["plot"]["x_min"],
        config["plot"]["x_max"],
        config["gaussian"]["number_of_centers"],
    )
    gaussian_width = config["gaussian"]["width"]

    sigmoid_centers = np.linspace(
        config["plot"]["x_min"],
        config["plot"]["x_max"],
        config["sigmoid"]["number_of_centers"],
    )
    sigmoid_width = config["sigmoid"]["width"]
    poly_matrix = polynomial_basis(
        x_train,
        degree=polynomial_degree,
    )
    gaussian_matrix = gaussian_basis(
        x_train,
        centers=gaussian_centers,
        width=gaussian_width,
    )
    sigmoid_matrix = sigmoidal_basis(
        x_train,
        centers=sigmoid_centers,
        width=sigmoid_width,
    )
    training_matrices = [poly_matrix, gaussian_matrix, sigmoid_matrix]
    plotting_matrices = [
        polynomial_basis(
            x,
            degree=polynomial_degree,
        ),
        gaussian_basis(
            x,
            centers=gaussian_centers,
            width=gaussian_width,
        ),
        sigmoidal_basis(
            x,
            centers=sigmoid_centers,
            width=sigmoid_width,
        ),
    ]
    titles = [
        "Polynomial basis",
        "Gaussian basis",
        "Sigmoidal basis",
    ]
    figure, axes = plt.subplots(
        3,
        1,
        sharex=True,
        sharey=True,
    )
    for axis, training_matrix, plotting_matrix, title in zip(
        axes.flat,
        training_matrices,
        plotting_matrices,
        titles,
    ):
        model = SequentialLinearRegression(
            learning_rate=config["learning_rate"], epochs=config["data"]["sample_size"]
        )
        model.fit(design_matrix=training_matrix, targets=t_train)
        prediction = model.predict(design_matrix=plotting_matrix)
        axis.plot(
            x,
            true_values,
            color="green",
            label="Ground truth",
        )
        axis.plot(
            x,
            prediction,
            color="red",
            label="Fitted model",
        )
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
    figure.tight_layout()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_sequential_learning.yaml")

    rng = create_rng(seed=config["seed"])
    x_train, t_train = create_data(
        sample_size=config["data"]["sample_size"],
        std=config["data"]["std"],
        rng=rng,
        domain=[0, 1],
    )

    figure = create_figure(x_train=x_train, t_train=t_train, config=config)

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
