import numpy as np
from matplotlib import pyplot as plt

from prml.distributions import Beta
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_figure(parameters, grid_points, y_max):
    mu = np.linspace(0, 1, grid_points)
    figure, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for axis, values in zip(axes.flat, parameters):
        a = values["a"]
        b = values["b"]
        density = Beta(a=a, b=b).pdf(mu)
        axis.plot(
            mu,
            density,
            color="red",
        )

        axis.text(
            0.10,
            0.87,
            rf"$a={a}$",
            transform=axis.transAxes,
        )
        axis.text(
            0.10,
            0.74,
            rf"$b={b}$",
            transform=axis.transAxes,
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, y_max),
            xlabel=r"$\mu$",
        )
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xticklabels(["0", "0.5", "1"])
        axis.set_yticks([0.0, 1.0, 2.0, 3.0])
    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_2.yaml"
    config = load_config(config_path)
    figure = create_figure(
        parameters=config["parameters"],
        grid_points=config["grid_points"],
        y_max=config["y_max"],
    )
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close()
    return output


def main():
    run()


if __name__ == "__main__":
    main()
