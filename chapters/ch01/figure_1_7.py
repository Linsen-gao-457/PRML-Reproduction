import numpy as np
from matplotlib import pyplot as plt

from prml.linear_models import RidgeRegression, polynomial_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng

from .figure_1_2 import generate_data, target_function


def create_figure(log_regularization_values, grid_points: int, degree, x, targets):
    grid = np.linspace(0, 1.0, grid_points)
    train_design_matrix = polynomial_basis(x, degree)
    grid_design_matrix = polynomial_basis(grid, degree)
    figure, axes = plt.subplots(
        1, len(log_regularization_values), sharex=True, sharey=True
    )
    for axis, log_regularization_value in zip(axes, log_regularization_values):
        regularization = np.exp(log_regularization_value)

        model = RidgeRegression(regularization)
        model.fit(design_matrix=train_design_matrix, targets=targets)
        predictions = model.predict(design_matrix=grid_design_matrix)
        axis.plot(grid, predictions, color="red")
        axis.plot(grid, target_function(grid), color="green")
        axis.scatter(x, targets, facecolor="none", edgecolor="blue")
        axis.text(
            0.59,
            0.79,
            rf"$\ln\lambda={log_regularization_value}$",
            transform=axis.transAxes,
        )
        axis.set(
            xlim=(-0.05, 1.05),
            ylim=(-1.6, 1.6),
            xticks=[0, 1],
            yticks=[-1, 0, 1],
            xlabel="$x$",
            ylabel="$t$",
        )
    return figure


def run(config_path=None):
    config_path = config_path or CONFIGS_DIR / "ch01/figure_1_7.yaml"
    config = load_config(config_path)
    rng = create_rng(config["seed"])

    x, targets = generate_data(
        sample_size=config["sample_size"],
        noise_std=config["noise_std"],
        rng=rng,
    )

    figure = create_figure(
        x=x,
        targets=targets,
        degree=config["degree"],
        log_regularization_values=config["log_regularization_values"],
        grid_points=config["grid_points"],
    )

    output = save_figure(
        figure,
        PROJECT_ROOT / config["output"],
    )
    plt.close(figure)

    print(f"Saved: {output}")
    return output


def main():
    run()


if __name__ == "__main__":
    main()
