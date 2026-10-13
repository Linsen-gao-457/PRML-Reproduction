import matplotlib.pyplot as plt
import numpy as np

from prml.distributions.multivariate_gaussian import MultivariateGaussian
from prml.distributions.uniform import Uniform
from prml.linear_models.bayesian_linear_regression import BayesianLinearRegression
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config, rng):
    config = config["data"]
    x_distribution = Uniform(
        low=config["x_min"],
        high=config["x_max"],
    )
    x = x_distribution.draw(
        config["sample_size"],
        rng,
    )
    x = np.stack([np.ones_like(x), x], axis=1)
    weights_true = np.asarray(
        [
            config["bias"],
            config["slope"],
        ]
    )
    t = np.dot(x, weights_true)
    t += rng.normal(scale=config["noise_std"], size=x.size)
    return x, t, weights_true


def normalized_density(distribution, grid):
    points = grid.reshape(-1, 2)
    density = distribution.pdf(points)
    maximum = density.max()
    density = density.reshape(grid.shape[:2])
    if maximum == 0:
        return density
    return density / maximum


def draw_weight_space(axis, grids, density, weight_values, config, weights_true):
    axis.contourf(
        (weight_values, weight_values),
        density,
        levels=config["plot"]["contour_levels"],
        cmap="jet",
    )
    axis.plot(
        weights_true[0],
        weights_true[1],
        marker="+",
        color="white",
        markersize=8,
        markeredgewidth=1.5,
    )
    axis.set_xlabel(r"$w_0$")
    axis.set_ylabel(
        r"$w_1$",
        rotation=0,
    )
    axis.set_xlim(-1, 1)

    axis.set_ylim(-1, 1)
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )


def draw_data_space(axis, model, observed_x, observed_t, config, rng):
    data_config = config["data"]
    plot_config = config["plot"]
    x = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        plot_config["line_points"],
    )
    sample_weights = model.sample_weights(
        sample_size=config["model"]["weight_samples"], rng=rng
    )
    x_0 = np.stack(
        (
            np.ones_like(x.size),
            x,
        ),
        axis=1,
    )
    for weight in sample_weights:
        y = np.dot(x_0, weight)
        axis.plot(
            x,
            y,
            color="red",
        )
    if observed_x.size > 0:
        axis.scatter(
            observed_x,
            observed_t,
            facecolors="none",
            edgecolors="blue",
        )
    axis.set_xlim(
        data_config["x_min"],
        data_config["x_max"],
    )

    axis.set_ylim(
        plot_config["target_min"],
        plot_config["target_max"],
    )

    axis.set_xlabel(r"$x$")
    axis.set_ylabel(
        r"$y$",
        rotation=0,
    )
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )


def normalized_likelihood(
    grid,
    x,
    target,
    noise_precision,
):
    pass


def create_figure(config, rng):
    x_data, t_data, weights_true = create_data(config, rng)
    alpha = config["model"]["alpha"]
    precision = 1 / (config["data"]["noise_std"]) ** 2
    prior = MultivariateGaussian(mean=np.zeros(2), cov=np.eye(N=2) / alpha)
    model = BayesianLinearRegression(prior, precision)
    weight_values = np.linspace(
        config["plot"]["weight_min"],
        config["plot"]["weight_max"],
        config["plot"]["grid_points"],
    )

    weight_0_grid, weight_1_grid = np.meshgrid(
        weight_values,
        weight_values,
    )
    parameter_grid = np.stack((weight_0_grid, weight_1_grid), axis=-1)
    figure, axes = plt.subplots(
        4,
        3,
    )
    axes[0, 0].set_title("likelihood")
    axes[0, 0].axis("off")

    axes[0, 1].set_title("prior/posterior")
    axes[0, 2].set_title("data space")
    prior_density = normalized_density(model.prior, parameter_grid)
    draw_weight_space(
        axis=axes[0, 1],
        grids=parameter_grid,
        config=config,
        density=prior_density,
        weights_true=weights_true,
    )
    draw_data_space(
        axis=axes[0, 2],
        model=model,
        observed_x=[],
        observed_t=[],
        config=config,
        rng=rng,
    )
    displayed_rows = {
        1: 1,
        2: 2,
        20: 3,
    }
    for count, (x, t) in enumerate(zip(x_data, t_data), start=1):
        likelihood = normalized_likelihood(
            grid=parameter_grid,
            x=x,
            target=t,
            noise_precision=(model.noise_precision),
        )
        basis = np.array([1.0, x])
        model.udpate(basis=basis, target=t)
        if count not in displayed_rows:
            continue
        row = displayed_rows[count]
        draw_weight_space(
            axis=axes[row, 0],
            weights_true=weight_values,
            density=likelihood,
            true_weights=weights_true,
            config=config,
        )
        posterior_density = normalized_density(
            distribution=model.posterior,
            grid=parameter_grid,
        )

        draw_weight_space(
            axis=axes[row, 1],
            weights_true=weight_values,
            density=posterior_density,
            true_weights=weights_true,
            config=config,
        )

        draw_data_space(
            axis=axes[row, 2],
            model=model,
            observed_x=x_data[:count],
            observed_t=t[:count],
            config=config,
            rng=rng,
        )

    figure.tight_layout()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_3_7.yaml")

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
