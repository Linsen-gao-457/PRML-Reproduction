"""Placeholder entry point for the polynomial curve-fitting comparison."""

import numpy as np
from matplotlib import pyplot as plt

from prml.linear_models import LinearRegression, polynomial_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng

from .figure_1_2 import generate_data, target_function


def fit_polynomial(x, targets, degree):
    """Fit w matrix of the polynomial by least squares."""
    design_matrix = polynomial_basis(x, degree)
    return LinearRegression().fit(design_matrix, targets=targets)


def create_figure(x: np.ndarray, t: np.ndarray, degrees: list[int]):
    grid = np.linspace(0, 1, 500)
    figure, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for axis, degree in zip(axes.flat, degrees):
        model = fit_polynomial(x, t, degree)
        predictions = model.predict(polynomial_basis(grid, degree))
        axis.plot(grid, target_function(grid), color="green")
        axis.scatter(
            x,
            t,
            facecolors="none",
            edgecolors="blue",
            label="training data",
        )
        axis.text(
            0.85,
            0.85,
            f"$M = {degree}$",
            transform=axis.transAxes,
            fontsize=12,
            ha="center",
            va="center",
        )
        axis.plot(grid, predictions, color="red")
        axis.set(xlim=(-0.05, 1.05), ylim=(-1.5, 1.5), xlabel="$x$", ylabel="$t$")
        axis.set_xticks([0, 1])
        axis.set_yticks([-1, 0, 1])
    return figure


def run():
    configure_path = CONFIGS_DIR / "ch01/figure_1_4.yaml"
    configure = load_config(configure_path)
    rng = create_rng(seed=configure["seed"])
    x, t = generate_data(
        sample_size=configure["sample_size"], noise_std=configure["noise_std"], rng=rng
    )
    figure = create_figure(x, t, degrees=configure["degrees"])
    output = save_figure(figure, PROJECT_ROOT / configure["output"])
    plt.close(figure)
    print(f"Saved: {output}")
    return output


def main():
    run()


if __name__ == "__main__":
    main()
