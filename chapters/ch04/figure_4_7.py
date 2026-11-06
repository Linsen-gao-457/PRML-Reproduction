import matplotlib.pyplot as plt
import numpy as np

from prml.classification import PerceptronClassifier
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config):
    pass


def draw_panel(
    axis,
    features,
    targets,
    weights,
    config,
    update_record=None,
):
    pass


def create_figure(config):
    features, targets = create_data(config)
    model = PerceptronClassifier()
    model.fit(featrues=features, targets=targets)
    if len(model.history_) != 2:
        raise RuntimeError(
            "Figure 4.7 expects exactly two perceptron updates, "
            f"but observed {len(model.history_)}"
        )
    first_update = model.history_[0]
    second_update = model.history_[1]
    figure, axes = plt.subplots(nrows=2, ncols=2)
    draw_panel(
        axis=axes[0, 0],
        features=features,
        targets=targets,
        weights=first_update["weights_before"],
        config=config,
    )
    draw_panel(
        axis=axes[0, 1],
        features=features,
        targets=targets,
        weights=first_update["weights_after"],
        config=config,
    )
    draw_panel(
        axis=axes[1, 0],
        features=features,
        targets=targets,
        weights=second_update["weights_before"],
        update_record=second_update,
        config=config,
    )
    draw_panel(
        axis=axes[1, 1],
        features=features,
        targets=targets,
        weights=second_update["weights_after"],
        config=config,
    )

    return figure


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
