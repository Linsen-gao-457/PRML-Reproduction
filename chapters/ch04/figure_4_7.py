import matplotlib.pyplot as plt
import numpy as np

from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_figure():
    pass


def run():
    config = load_config(CONFIGS_DIR / "ch04/figure_4_7.yaml")

    rng = create_rng(seed=config["seed"])

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
