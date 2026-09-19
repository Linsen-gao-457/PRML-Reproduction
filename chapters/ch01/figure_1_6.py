import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models import LinearRegression, polynomial_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng

from .figure_1_2 import generate_data, target_function
from .figure_1_4 import fit_polynomial


def create_figure(datasets, degree):
    grid = np.linspace(0, 1, 500)
    grad_designed_matrix = polynomial_basis(x=grid, degree=degree)
    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for axis, (sample_size, x, t) in zip(axes, datasets):
        model = fit_polynomial(x, t, degree)
        predictions = model.predict(grad_designed_matrix)
        axis.plot(
            grid, target_function(grid), color="green", label="groundtruth function"
        )
        axis.plot(grid, predictions, color="red", label="Prediction")
        axis.scatter(
            x,
            t,
            color="blue",
            facecolors="none",
            label="Train",
        )
        axis.set(
            xlim=(-0.05, 1.05),
            ylim=(-1.6, 1.6),
            xticks=[0, 1],
            yticks=[-1, 0, 1],
            xlabel="$x$",
            ylabel="$t$",
        )

        axis.text(
            0.77,
            0.82,
            rf"$N={sample_size}$",
            transform=axis.transAxes,
            fontsize=18,
        )

    return figure


def run():
    configure_path = CONFIGS_DIR / "ch01/figure_1_6.yaml"
    config = load_config(configure_path)
    rng = create_rng(config["seed"])

    datasets = []
    for sample_size in config["sample_sizes"]:
        x, t = generate_data(
            sample_size=sample_size, noise_std=config["noise_std"], rng=rng
        )
        datasets.append((sample_size, x, t))

    figure = create_figure(datasets=datasets, degree=config["degree"])
    output = save_figure(figure, path=PROJECT_ROOT / config["output"])
    plt.close(figure)
    return output


def main():
    run()


if __name__ == "__main__":
    main()
