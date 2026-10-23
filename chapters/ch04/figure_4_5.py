import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from prml.classification.least_squares_classifier import (
    LeastSquaresClassifier,
)
from prml.classification.multiclass_logistic_regression import (
    MulticlassLogisticRegression,
)
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config, rng):
    data_config = config["data"]
    covariance = np.asarray(
        data_config["covariance"],
        dtype=float,
    )

    feature_groups = []
    target_groups = []

    for class_index, mean in enumerate(data_config["class_means"]):
        class_features = rng.multivariate_normal(
            mean=mean,
            cov=covariance,
            size=data_config["class_size"],
        )

        feature_groups.append(class_features)
        target_groups.append(
            np.full(
                data_config["class_size"],
                class_index,
                dtype=int,
            )
        )

    features = np.vstack(feature_groups)
    targets = np.concatenate(target_groups)

    return features, targets


def create_prediction_grid(config):
    plot_config = config["plot"]

    x_values = np.linspace(
        plot_config["x_min"],
        plot_config["x_max"],
        plot_config["grid_points"],
    )
    y_values = np.linspace(
        plot_config["y_min"],
        plot_config["y_max"],
        plot_config["grid_points"],
    )

    x_grid, y_grid = np.meshgrid(
        x_values,
        y_values,
    )
    grid_features = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
        )
    )

    return x_grid, y_grid, grid_features


def predict_least_squares(model, features):
    scores = model.decision_function(features)
    class_indices = np.argmax(scores, axis=1)
    return model.classes_[class_indices]


def draw_points(axis, features, targets, config):
    plot_config = config["plot"]

    class_styles = (
        {
            "color": plot_config["red_color"],
            "marker": "x",
        },
        {
            "color": plot_config["green_color"],
            "marker": "+",
        },
        {
            "color": plot_config["blue_color"],
            "marker": "o",
        },
    )

    for class_index, style in enumerate(class_styles):
        class_features = features[targets == class_index]

        if style["marker"] == "o":
            axis.scatter(
                class_features[:, 0],
                class_features[:, 1],
                marker="o",
                facecolors="none",
                edgecolors=style["color"],
                s=plot_config["marker_size"],
                linewidths=plot_config["marker_line_width"],
                zorder=3,
            )
        else:
            axis.scatter(
                class_features[:, 0],
                class_features[:, 1],
                marker=style["marker"],
                color=style["color"],
                s=plot_config["marker_size"],
                linewidths=plot_config["marker_line_width"],
                zorder=3,
            )


def draw_decision_regions(
    axis,
    features,
    targets,
    model,
    prediction_function,
    title,
    config,
):
    plot_config = config["plot"]

    x_grid, y_grid, grid_features = create_prediction_grid(config)

    predictions = prediction_function(
        model,
        grid_features,
    )
    prediction_indices = np.searchsorted(
        model.classes_,
        predictions,
    )
    prediction_grid = prediction_indices.reshape(x_grid.shape)

    background_colors = ListedColormap(plot_config["background_colors"])

    axis.contourf(
        x_grid,
        y_grid,
        prediction_grid,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=background_colors,
        alpha=plot_config["background_alpha"],
    )

    axis.contour(
        x_grid,
        y_grid,
        prediction_grid,
        levels=[0.5, 1.5],
        colors="black",
        linewidths=plot_config["boundary_line_width"],
    )

    draw_points(
        axis,
        features,
        targets,
        config,
    )

    axis.set_title(title)
    axis.set_xlim(
        plot_config["x_min"],
        plot_config["x_max"],
    )
    axis.set_ylim(
        plot_config["y_min"],
        plot_config["y_max"],
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(plot_config["ticks"])
    axis.set_yticks(plot_config["ticks"])
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )


def create_figure(config, rng):
    features, targets = create_data(
        config,
        rng,
    )

    least_squares = LeastSquaresClassifier()
    least_squares.fit(
        features,
        targets,
    )

    logistic_regression = MulticlassLogisticRegression(
        tolerance=config["model"]["tolerance"],
        max_iterations=config["model"]["max_iterations"],
    )
    logistic_regression.fit(
        features,
        targets,
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=config["plot"]["figure_size"],
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    draw_decision_regions(
        axis=axes[0],
        features=features,
        targets=targets,
        model=least_squares,
        prediction_function=predict_least_squares,
        title="Least-squares classifier",
        config=config,
    )

    draw_decision_regions(
        axis=axes[1],
        features=features,
        targets=targets,
        model=logistic_regression,
        prediction_function=lambda model, points: model.predict(points),
        title="Multiclass logistic regression",
        config=config,
    )

    axes[0].set_xlabel(r"$x_1$")
    axes[1].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(
        r"$x_2$",
        rotation=0,
    )

    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch04/figure_4_5.yaml")

    rng = create_rng(seed=config["seed"])

    figure = create_figure(
        config=config,
        rng=rng,
    )

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
