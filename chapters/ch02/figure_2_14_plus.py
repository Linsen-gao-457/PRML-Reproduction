import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import (
    GaussianMeanPrecisionBayes,
    NormalGamma,
)
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def evaluate_density(distribution, mu_grid, precision_grid):
    points = np.column_stack(
        (
            mu_grid.ravel(),
            precision_grid.ravel(),
        )
    )

    density = distribution.pdf(points)
    density = density.reshape(mu_grid.shape)

    # Relative density makes the same contour levels useful in every panel.
    maximum = density.max()
    if maximum > 0:
        density = density / maximum

    return density


def create_figure(
    observations,
    prior,
    sample_sizes,
    true_mean,
    true_precision,
    mu_min,
    mu_max,
    precision_min,
    precision_max,
    mu_grid_points,
    precision_grid_points,
    contour_levels,
):
    mu = np.linspace(
        mu_min,
        mu_max,
        mu_grid_points,
    )

    precision = np.linspace(
        precision_min,
        precision_max,
        precision_grid_points,
    )

    mu_grid, precision_grid = np.meshgrid(
        mu,
        precision,
    )

    distributions = [(0, prior)]

    model = GaussianMeanPrecisionBayes(prior=prior)

    for sample_size in sample_sizes:
        posterior = model.fit(observations[:sample_size])
        distributions.append((sample_size, posterior))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
    )

    for axis, (sample_size, distribution) in zip(
        axes.flat,
        distributions,
    ):
        density = evaluate_density(
            distribution=distribution,
            mu_grid=mu_grid,
            precision_grid=precision_grid,
        )

        axis.contour(
            mu_grid,
            precision_grid,
            density,
            levels=contour_levels,
            colors="blue",
        )

        axis.scatter(
            true_mean,
            true_precision,
            color="red",
            marker="x",
            s=60,
            label="true value",
        )

        axis.set_title(rf"$N={sample_size}$")
        axis.set(
            xlim=(mu_min, mu_max),
            ylim=(precision_min, precision_max),
            xlabel=r"mean $\mu$",
            ylabel=r"precision $\lambda$",
        )

        axis.tick_params(
            direction="in",
            top=True,
            right=True,
        )

    axes[0, 0].legend(frameon=False)

    figure.tight_layout()
    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_14.yaml"
    config = load_config(config_path)

    rng = create_rng(seed=config["seed"])

    true_mean = config["data"]["mean"]
    true_variance = config["data"]["variance"]
    true_precision = 1.0 / true_variance

    observations = rng.normal(
        loc=true_mean,
        scale=np.sqrt(true_variance),
        size=config["data"]["data_sample_size"],
    )

    prior = NormalGamma.form_constraints(
        c=config["prior"]["c"],
        beta=config["prior"]["beta"],
        d=config["prior"]["d"],
    )

    figure = create_figure(
        observations=observations,
        prior=prior,
        sample_sizes=config["posterior_sample_sizes"],
        true_mean=true_mean,
        true_precision=true_precision,
        mu_min=config["plot"]["mu_min"],
        mu_max=config["plot"]["mu_max"],
        precision_min=config["plot"]["precision_min"],
        precision_max=config["plot"]["precision_max"],
        mu_grid_points=config["plot"]["mu_grid_points"],
        precision_grid_points=config["plot"]["precision_grid_points"],
        contour_levels=config["plot"]["contour_levels"],
    )

    output = save_figure(
        figure,
        PROJECT_ROOT / config["output"],
    )

    plt.close(figure)
    return output


def main():
    output = run()
    return output


if __name__ == "__main__":
    main()
