import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models.basis_functions import (
    gaussian_basis,
    polynomial_basis,
    sigmoidal_basis,
)
from prml.linear_models.ridge_regression import RidgeRegression
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(sample_size, std, domain: list[float], rng):
    x = np.linspace(domain[0], domain[1], sample_size)
    t = np.sin(2 * np.pi * x) + rng.normal(scale=std, size=sample_size)
    return x, t


def sinusoidal(x):
    return np.sin(2 * np.pi * x)


def create_figure(config, rng, x_train, t_train):
    pass


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_ridge_regression.yaml")

    rng = create_rng(seed=config["seed"])

    x_train, t_train = create_data(
        sample_size=config["data"]["sample_size"],
        std=config["data"]["std"],
        rng=rng,
        domain=[config["plot"]["x_min"], config["plot"]["x_max"]],
    )
    figure = create_figure()

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
