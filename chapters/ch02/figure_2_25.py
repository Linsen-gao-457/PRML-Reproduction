import matplotlib.pyplot as plt
import numpy as np

from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def gaussian_pdf(
    x,
    mean,
    standard_deviation,
):
    normalization = 1.0 / (np.sqrt(2.0 * np.pi) * standard_deviation)

    exponent = np.exp(-0.5 * ((x - mean) / standard_deviation) ** 2)

    return normalization * exponent


def mixture_pdf(x, mixture):
    density = np.zeros_like(
        x,
        dtype=float,
    )

    for component in mixture:
        density += component["weight"] * gaussian_pdf(
            x=x,
            mean=component["mean"],
            standard_deviation=component["standard_deviation"],
        )

    return density


def draw_mixture_samples(
    sample_size,
    mixture,
    rng,
):
    weights = np.array(
        [component["weight"] for component in mixture],
        dtype=float,
    )

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("mixture weights must sum to one")

    component_indices = rng.choice(
        len(mixture),
        size=sample_size,
        p=weights,
    )

    means = np.array([component["mean"] for component in mixture])

    standard_deviations = np.array(
        [component["standard_deviation"] for component in mixture]
    )

    return rng.normal(
        loc=means[component_indices],
        scale=standard_deviations[component_indices],
    )


def kernel_density(
    x,
    observations,
    bandwidth,
):
    # Shape:
    # x[:, None]              -> (grid_points, 1)
    # observations[None, :]   -> (1, sample_size)
    # difference              -> (grid_points, sample_size)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    difference = x[:, None] - observations[None, :]
    normalization = np.sqrt(2.0 * np.pi) * bandwidth
    kernels = np.exp(-0.5 * (difference / bandwidth) ** 2)
    return np.mean(kernels / normalization, axis=1)


def create_figure(observations, config):
    plot = config["plot"]

    x = np.linspace(
        start=plot["x_min"],
        stop=plot["x_max"],
        num=plot["grid_points"],
    )
    true_density = mixture_pdf(x=x, mixture=config["mixture"])
    bandwidths = config["kernel"]["bandwidths"]
    figure, axes = plt.subplots(
        len(bandwidths),
        1,
        figsize=(
            plot["figure_width"],
            plot["figure_height"],
        ),
        sharex=True,
        sharey=True,
    )
    for axis, bandwidth in zip(axes, bandwidths):
        estimated_density = kernel_density(
            x=x, observations=observations, bandwidth=bandwidth
        )
        axis.plot(
            x,
            estimated_density,
            color=config["kernel"]["color"],
            linewidth=config["kernel"]["line_width"],
        )
        axis.plot(
            x,
            true_density,
            color=config["true_density"]["color"],
            linewidth=config["true_density"]["line_width"],
        )

        axis.text(
            0.05,
            0.78,
            rf"$h={bandwidth:g}$",
            transform=axis.transAxes,
            fontsize=12,
        )
        axis.set(
            xlim=(
                plot["x_min"],
                plot["x_max"],
            ),
            ylim=(
                plot["y_min"],
                plot["y_max"],
            ),
        )

        axis.set_yticks(
            [
                plot["y_min"],
                plot["y_max"],
            ]
        )

        axis.tick_params(
            direction="in",
            top=True,
            right=True,
        )

    axes[-1].set_xticks(
        [
            plot["x_min"],
            0.5,
            plot["x_max"],
        ]
    )

    axes[-1].set_xlabel(r"$x$")

    figure.tight_layout()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch02/figure_2_25.yaml")

    rng = create_rng(seed=config["seed"])

    observations = draw_mixture_samples(
        sample_size=config["sample_size"],
        mixture=config["mixture"],
        rng=rng,
    )

    figure = create_figure(
        observations=observations,
        config=config,
    )

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
