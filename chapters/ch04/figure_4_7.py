import matplotlib.pyplot as plt
import numpy as np

from prml.classification import PerceptronClassifier
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config):
    data_config = config["data"]
    features = np.asarray(
        data_config["features"],
        dtype=float,
    )
    targets = np.asarray(
        data_config["targets"],
        dtype=int,
    )
    return features, targets


def draw_points(axis, features, targets, config):
    plot_config = config["plot"]
    blue_points = features[targets == 1]
    red_points = features[targets == 0]
    axis.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        color=plot_config["blue_color"],
        s=plot_config["marker_size"],
        zorder=3,
    )
    axis.scatter(
        red_points[:, 0],
        red_points[:, 1],
        color=plot_config["red_color"],
        s=plot_config["marker_size"],
        zorder=3,
    )


def draw_decision_boundary(axis, weights, config):
    plot_config = config["plot"]

    if np.isclose(weights[1], 0.0):
        axis.axvline(
            0.0,
            color=plot_config["boundary_color"],
            linewidth=plot_config["boundary_line_width"],
        )
        return

    x = np.linspace(
        plot_config["x_min"],
        plot_config["x_max"],
        plot_config["line_points"],
    )
    y = -(weights[0] / weights[1]) * x

    axis.plot(
        x,
        y,
        color=plot_config["boundary_color"],
        linewidth=plot_config["boundary_line_width"],
        zorder=1,
    )


def draw_arrow(
    axis,
    start,
    end,
    color,
    config,
    zorder=4,
):
    plot_config = config["plot"]

    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": plot_config["arrow_line_width"],
            "mutation_scale": plot_config["arrow_head_size"],
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=zorder,
    )


def draw_weight_vector(axis, weights, config):
    draw_arrow(
        axis=axis,
        start=np.zeros(2),
        end=weights,
        color=config["plot"]["weight_color"],
        config=config,
    )


def draw_update(
    axis,
    update_record,
    config,
):
    plot_config = config["plot"]
    feature = update_record["feature"]
    weights_before = update_record["weights_before"]
    weights_after = update_record["weights_after"]

    draw_arrow(
        axis=axis,
        start=np.zeros(2),
        end=feature,
        color=plot_config["update_color"],
        config=config,
        zorder=4,
    )
    draw_arrow(
        axis=axis,
        start=weights_before,
        end=weights_after,
        color=plot_config["update_color"],
        config=config,
        zorder=5,
    )

    axis.scatter(
        [feature[0]],
        [feature[1]],
        s=plot_config["highlight_size"],
        facecolors="none",
        edgecolors=plot_config["highlight_color"],
        linewidths=plot_config["highlight_line_width"],
        zorder=6,
    )


def configure_axis(axis, config):
    plot_config = config["plot"]

    axis.set_xlim(
        plot_config["x_min"],
        plot_config["x_max"],
    )
    axis.set_ylim(
        plot_config["y_min"],
        plot_config["y_max"],
    )
    axis.set_xticks(plot_config["ticks"])
    axis.set_yticks(plot_config["ticks"])
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )


def draw_panel(
    axis,
    features,
    targets,
    weights,
    config,
    update_record=None,
):
    draw_points(axis, features, targets, config)
    draw_decision_boundary(axis, weights, config)
    draw_weight_vector(axis, weights, config)
    if update_record is not None:
        draw_update(
            axis,
            update_record,
            config,
        )
    configure_axis(axis, config)


def create_figure(config):
    features, targets = create_data(config)
    model = PerceptronClassifier()
    model.fit(
        featrues=features,
        targets=targets,
        initial_weight=config["model"]["initial_weights"],
    )
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
        update_record=first_update,
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

    figure = create_figure(config)

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
