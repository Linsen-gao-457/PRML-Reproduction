import matplotlib.pyplot as plt
import numpy as np

from prml.classification import FisherLinearDiscriminant
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config, rng):
    pass


def class_means(features, targets):
    pass


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm


def draw_data(axis, features, targets, config):
    pass


def configure_axis(axis, config):
    pass


def draw_prjection_histograms(
    axis,
    features,
    targets,
    calss_mean_values,
    direction,
    anchor,
    config,
):
    pass


def create_figure(config, rng):
    features, targets = create_data(config=config, rng=rng)
    means = class_means(features=features, targets=targets)
    mean_direction_unit = normalize(means[1] - means[0])
    model = FisherLinearDiscriminant()
    model.fit(features, targets)
    figure, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    for axis in axes:
        draw_data(axis=axis, features=features, targets=targets, config=config)
        configure_axis(axis=axis, config=config)


def run():
    config = load_config(CONFIGS_DIR / "ch04/figure_4_6.yaml")

    rng = create_rng(seed=config["seed"])

    figure = create_figure(config=config, rng=rng)

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
