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
    density = np.zeros_like(x, dtype=float)
    for component in mixture:
        density += component["weight"] * gaussian_pdf(
            x=x,
            mean=component["mean"],
            standard_deviation=component["standard_deviation"],
        )
    return density


def draw_mixture_samples(sample_size, mixture, rng):
    weights = np.array([component["weight"] for component in mixture], dtype=float)
    if not np.isclose(weights.sum(), 1):
        raise ValueError("mixture weight must sum to 1")
    component_indices = rng.choice(len(mixture), size=sample_size, p=weights)
    means = np.array([component["mean"] for component in mixture], dtype=float)
    standard_deviations = np.array(
        [component["standard_deviation"] for component in mixture]
    )
    return rng.normal(
        loc=means[component_indices],
        scale=standard_deviations[component_indices],
    )


def histogram_density(
    observations,
    bin_width,
    x_min,
    x_max,
):
    bin_edges = np.arange(
        x_min,
        x_max + bin_width,
        bin_width,
    )

    counts, bin_edges = np.histogram(
        observations,
        bins=bin_edges,
    )

    density = counts / (observations.size * bin_width)

    return bin_edges, density


def create_figure(
    observations,
    mixture,
    bin_widths,
    histogram_color,
    histogram_edge_color,
    density_color,
    density_line_width,
    x_min,
    x_max,
    y_min,
    y_max,
    grid_points,
    figure_width,
    figure_height,
):
    x = np.linspace(x_min, x_max, grid_points)
    true_density = mixture_pdf(x, mixture=mixture)
    figure, axes = plt.subplots(
        len(bin_widths),
        1,
        figsize=(
            figure_width,
            figure_height,
        ),
        sharex=True,
        sharey=True,
    )
    for axis, bin_width in zip(
        axes.flat,
        bin_widths,
    ):
        bin_edges, estimated_density = histogram_density(
            observations=observations,
            bin_width=bin_width,
            x_min=x_min,
            x_max=x_max,
        )
        axis.bar(
            bin_edges[:-1],
            estimated_density,
            width=bin_width,
            align="edge",
            color=histogram_color,
            edgecolor=histogram_edge_color,
            linewidth=0.8,
        )
        axis.plot(
            x,
            true_density,
            color=density_color,
            linewidth=density_line_width,
        )

        axis.text(
            0.05,
            0.78,
            rf"$\Delta={bin_width:g}$",
            transform=axis.transAxes,
            fontsize=12,
        )

        axis.set(
            xlim=(x_min, x_max),
            ylim=(y_min, y_max),
        )

        axis.set_yticks([y_min, y_max])

        axis.tick_params(
            direction="in",
            top=True,
            right=True,
        )

    axes[-1].set_xticks([x_min, 0.5, x_max])

    axes[-1].set_xlabel(r"$x$")

    figure.tight_layout()
    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_24.yaml"
    config = load_config(config_path)

    rng = create_rng(seed=config["seed"])
    observations = draw_mixture_samples(
        sample_size=config["sample_size"],
        mixture=config["mixture"],
        rng=rng,
    )
    figure = create_figure(
        observations=observations,
        mixture=config["mixture"],
        bin_widths=config["histogram"]["bin_widths"],
        histogram_color=config["histogram"]["color"],
        histogram_edge_color=config["histogram"]["edge_color"],
        density_color=config["density_curve"]["color"],
        density_line_width=config["density_curve"]["line_width"],
        x_min=config["plot"]["x_min"],
        x_max=config["plot"]["x_max"],
        y_min=config["plot"]["y_min"],
        y_max=config["plot"]["y_max"],
        grid_points=config["plot"]["grid_points"],
        figure_width=config["plot"]["figure_width"],
        figure_height=config["plot"]["figure_height"],
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
