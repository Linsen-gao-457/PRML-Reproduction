import matplotlib

from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from prml.utils.config import load_config
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def target_function(x):
    return np.sin(2.0 * np.pi * np.asarray(x))


def generate_data(sample_size, noise_std, rng):
    x = np.linspace(0.0, 1.0, sample_size)
    targets = target_function(x) + rng.normal(0.0, noise_std, sample_size)
    return x, targets


def create_figure(x, targets):
    grid = np.linspace(0.0, 1.0, 200)
    figure, axis = plt.subplots()
    axis.plot(grid, target_function(grid), color="green", label="$\\sin(2\\pi x)$")
    axis.scatter(
        x,
        targets,
        facecolors="none",
        edgecolors="blue",
        label="training data",
    )
    axis.set(xlabel="$x$", ylabel="$t$", xlim=(-0.1, 1.1))
    axis.legend()
    return figure


def run(config_path=None):
    config_path = CONFIGS_DIR / "ch01/figure_1_2.yaml"
    config = load_config(config_path)
    rng = create_rng(config["seed"])
    x, targets = generate_data(
        sample_size=config["sample_size"],
        noise_std=config["noise_std"],
        rng=rng,
    )
    figure = create_figure(x, targets)
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close(figure)
    print(f"Saved: {output}")
    return output


def main():
    run()


if __name__ == "__main__":
    main()
