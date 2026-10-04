import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Gaussian, GaussianMeanBayes
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng

COLORS = {
    0: "black",
    1: "green",
    2: "blue",
    10: "red",
}


def create_figure(
    sample_dataset, prior, precision, grid_points, x_min, x_max, y_min, y_max
):
    pass


def run():
    configure_path = CONFIGS_DIR / "ch02/figure_2_12.yaml"
    config = load_config(configure_path)
    rng = create_rng(seed=config["seed"])
    sample_dataset = rng.normal(
        loc=config["data"]["mean"],
        scale=np.sqrt(
            config["data"]["variance"], size=config["data"]["data_sample_size"]
        ),
    )
    prior = Gaussian(
        mean=config["prior"]["mean"],
        standard_deviation=np.sqrt(config["prior"]["variance"]),
    )
    figure = create_figure(
        sample_dataset=sample_dataset,
        prior=prior,
        precision=1 / config["data"]["variance"],
        grid_points=config["plot"]["grid_points"],
        x_min=config["plot"]["x_min"],
        x_max=config["plot"]["x_max"],
        y_min=config["plot"]["y_min"],
        y_max=config["plot"]["y_max"],
    )
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close(figure)
    return output


def main():
    run()


if __name__ == "__main__":
    main()
