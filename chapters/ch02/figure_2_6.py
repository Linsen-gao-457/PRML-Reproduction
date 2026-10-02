import matplotlib.pyplot as plt
import numpy as np

from prml.distributions.uniform import Uniform
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_figure(
    rng, sample_sizes, num_experiments, x_min, x_max, y_max, seed, num_bins
):
    distribution = Uniform(low=x_min, high=x_max)
    x = np.linspace(0, 1, num_bins + 1)
    fig, axes = plt.subplots(1, len(sample_sizes), sharex=True, sharey=True)
    for axis, sample_size in zip(axes, sample_sizes):
        y = distribution.draw(rng=rng, sample_size=sample_size * num_experiments)
        y = y.reshape(num_experiments, -1)
        y = y.mean(axis=1)
        axis.hist(y, bins=x, color="blue", edgecolor="black", density=True)
        axis.text(0.04, 4.87, f"N={sample_size}")
        axis.set(xlim=(x_min, x_max), ylim=(0, y_max), xticks=[0, 0.5, 1])
    return fig


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_6.yaml"
    config = load_config(config_path)
    rng = create_rng(seed=config["seed"])
    figure = create_figure(
        rng=rng,
        sample_sizes=config["sample_sizes"],
        num_experiments=config["num_experiments"],
        x_min=config["x_min"],
        x_max=config["x_max"],
        y_max=config["y_max"],
        seed=config["seed"],
        num_bins=config["num_bins"],
    )
    output = save_figure(figure=figure, path=PROJECT_ROOT / config["output"])
    plt.close()
    return output


def main():
    run()


if __name__ == "__main__":
    main()
