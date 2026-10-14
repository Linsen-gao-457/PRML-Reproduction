import matplotlib.pyplot as plt
import numpy as np

from prml.distributions.multivariate_gaussian import MultivariateGaussian
from prml.linear_models.basis_functions import gaussian_basis
from prml.linear_models.bayesian_linear_regression import BayesianLinearRegression
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def sinusoidal(x):
    return np.sin(2.0 * np.pi * x)


def create_data(config, rng):
    data_config = config["data"]
    x = rng.uniform(
        low=data_config["x_min"],
        high=data_config["x_max"],
        size=max(data_config["sample_sizes"]),
    )
    targets = sinusoidal(x) + rng.normal(
        scale=data_config["noise_std"],
        size=x.size,
    )
    return x, targets


def draw_panel(axis, x_train, targets, x_plot, mean, std, config):
    plot_config = config["plot"]
    axis.fill_between(
        x_plot,
        mean - std,
        mean + std,
        color=plot_config["uncertainty_color"],
    )
    axis.plot(
        x_plot,
        sinusoidal(x_plot),
        color=plot_config["truth_color"],
    )
    axis.plot(
        x_plot,
        mean,
        color=plot_config["mean_color"],
    )
    axis.scatter(
        x_train,
        targets,
        facecolors="none",
        edgecolors=plot_config["data_color"],
        linewidths=1.5,
        s=36,
        zorder=3,
    )
    axis.set_xlim(config["data"]["x_min"], config["data"]["x_max"])
    axis.set_ylim(plot_config["target_min"], plot_config["target_max"])
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$t$", rotation=0)
    axis.set_xticks([0, 1])
    axis.set_yticks([-1, 0, 1])
    axis.tick_params(direction="in", top=True, right=True)


def create_figure(config, rng):
    data_config = config["data"]
    model_config = config["model"]
    plot_config = config["plot"]
    x_train, targets = create_data(config, rng)
    x_plot = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        plot_config["grid_points"],
    )
    centers = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        model_config["number_of_basis_functions"],
    )
    training_matrix = gaussian_basis(
        x_train,
        centers,
        model_config["basis_width"],
    )
    plotting_matrix = gaussian_basis(
        x_plot,
        centers,
        model_config["basis_width"],
    )
    prior = MultivariateGaussian(
        mean=np.zeros(centers.size),
        cov=np.eye(centers.size) / model_config["alpha"],
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=plot_config["figure_size"],
        sharex=True,
        sharey=True,
    )
    sample_sizes = data_config["sample_sizes"]
    sample_sizes_axes_dict = dict(zip(sample_sizes, axes.flat))
    model = BayesianLinearRegression(
        prior=prior,
        precision=1.0 / data_config["noise_std"] ** 2,
    )
    for sample_size, (basis, target) in enumerate(
        zip(training_matrix, targets), start=1
    ):
        model.update(basis=basis, target=target)
        if sample_size not in sample_sizes_axes_dict:
            continue
        predictive_mean, predictive_variance = model.predictive_distribution(
            design_matrix=plotting_matrix,
        )
        mean, std = (predictive_mean, np.sqrt(predictive_variance))
        draw_panel(
            sample_sizes_axes_dict[sample_size],
            x_train[:sample_size],
            targets[:sample_size],
            x_plot,
            mean,
            std,
            config,
        )
    figure.tight_layout()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_3_8.yaml")
    rng = create_rng(seed=config["seed"])
    figure = create_figure(config, rng)
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close(figure)
    return output


def main():
    run()


if __name__ == "__main__":
    main()
