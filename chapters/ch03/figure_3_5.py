import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models.basis_functions import gaussian_basis
from prml.linear_models.ridge_regression import RidgeRegression
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def sinusoidal(x):
    return np.sin(2.0 * np.pi * x)


def predict_datasets(
    training_matrix,
    plotting_matrix,
    datasets,
    regularization,
):
    return np.asarray(
        [
            RidgeRegression(regularization)
            .fit(training_matrix, targets)
            .predict(plotting_matrix)
            for targets in datasets
        ]
    )


def create_figure(config, rng):
    plot_config = config["plot"]
    data_config = config["data"]
    model_config = config["model"]
    x_train = np.linspace(
        plot_config["x_min"],
        plot_config["x_max"],
        data_config["sample_size"],
    )
    x_plot = np.linspace(
        plot_config["x_min"],
        plot_config["x_max"],
        plot_config["grid_points"],
    )
    centers = np.linspace(
        plot_config["x_min"],
        plot_config["x_max"],
        model_config["number_of_centers"],
    )
    datasets = sinusoidal(x_train)[None, :] + rng.normal(
        scale=data_config["noise_std"],
        size=(
            data_config["number_of_datasets"],
            data_config["sample_size"],
        ),
    )
    training_matrix = gaussian_basis(
        x=x_train, centers=centers, width=model_config["width"]
    )
    plotting_matrix = gaussian_basis(
        x_plot,
        centers,
        model_config["width"],
    )
    log_lambdas = model_config["log_regularization_values"]
    figure, axes = plt.subplots(
        len(log_lambdas),
        2,
        sharex=True,
        sharey=True,
    )
    for row, log_lambda in enumerate(log_lambdas):
        predictions = predict_datasets(
            training_matrix,
            plotting_matrix,
            datasets,
            regularization=np.exp(log_lambda),
        )
        left_axis, right_axis = axes[row]
        for prediction in predictions[: plot_config["number_of_displayed_fits"]]:
            left_axis.plot(x_plot, prediction, color="red", linewidth=0.4)
            left_axis.text(
                0.60,
                0.75,
                rf"$\ln\lambda={log_lambda:g}$",
                transform=left_axis.transAxes,
            )

            right_axis.plot(
                x_plot,
                sinusoidal(x_plot),
                color="green",
            )
            right_axis.plot(
                x_plot,
                predictions.mean(axis=0),
                color="red",
            )
    figure.tight_layout()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_3_5.yaml")

    rng = create_rng(seed=config["seed"])

    figure = create_figure(config, rng)

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
