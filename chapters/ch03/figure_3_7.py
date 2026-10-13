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


def draw_weight_space(axis, grids, density, values, config, weights_true):
    pass


def draw_data_space(axis, model, observed_x, observed_y, config, rng):
    pass


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
        observed_y=[],
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
            values=weight_values,
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
            values=weight_values,
            density=posterior_density,
            true_weights=weights_true,
            config=config,
        )

        draw_data_space(
            axis=axes[row, 2],
            model=model,
            observed_x=x_data[:count],
            observed_targets=t[:count],
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
