import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models.basis_functions import (
    gaussian_basis,
    polynomial_basis,
    sigmoidal_basis,
)
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_figure(config):
    x = np.linspace(
        config["plot"]["x_min"], config["plot"]["x_max"], config["plot"]["grid_points"]
    )
    polynomial_values = polynomial_basis(x=x, degree=config["polynomial"]["degree"])
    gaussian_centers = np.linspace(
        config["plot"]["x_min"],
        config["plot"]["x_max"],
        config["gaussian"]["number_of_centers"],
    )
    gaussian_values = gaussian_basis(
        x=x, centers=gaussian_centers, width=config["gaussian"]["width"]
    )
    sigmoid_centers = np.linspace(
        config["plot"]["x_min"],
        config["plot"]["x_max"],
        config["sigmoid"]["number_of_centers"],
    )
    sigmoid_values = sigmoidal_basis(
        x=x, centers=sigmoid_centers, width=config["sigmoid"]["width"]
    )
    figure, axes = plt.subplots(1, 3)
    basis_groups = [
        polynomial_values,
        gaussian_values,
        sigmoid_values,
    ]
    titles = [
        "Polynomial",
        "Gaussian",
        "Sigmoid",
    ]
    for axis, values, title in zip(axes, basis_groups, titles):
        for col in range(values.shape[1]):
            axis.plot(x, values[:, col])
            axis.set_xlim(
                config["plot"]["x_min"],
                config["plot"]["x_max"],
            )

        axis.set_title(title)

        axis.tick_params(
            direction="in",
            top=True,
            right=True,
        )
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_3_1.yaml")

    figure = create_figure(config=config)

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
