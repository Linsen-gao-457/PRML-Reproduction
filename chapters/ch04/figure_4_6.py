import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from prml.classification import FisherLinearDiscriminant
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config, rng):
    data_config = config["data"]
    covariance = np.asarray(data_config["covariance"], dtype=float)
    blue_points = rng.multivariate_normal(
        mean=data_config["blue_mean"],
        cov=covariance,
        size=data_config["class_size"],
    )
    red_points = rng.multivariate_normal(
        mean=data_config["red_mean"],
        cov=covariance,
        size=data_config["class_size"],
    )
    features = np.concatenate((blue_points, red_points), axis=0)
    targets = np.concatenate(
        (
            np.zeros(blue_points.shape[0], dtype=int),
            np.ones(red_points.shape[0], dtype=int),
        )
    )
    return features, targets


def class_means(features, targets):
    classes = np.unique(targets)
    return np.vstack(
        ([features[targets == class_label].mean(axis=0) for class_label in classes]),
    )


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm


def draw_data(axis, features, targets, config):
    plot_config = config["plot"]
    blue_points = features[targets == 0]
    red_points = features[targets == 1]
    axis.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        color=plot_config["blue_color"],
        s=plot_config["marker_size"],
        marker=".",
        zorder=3,
    )
    axis.scatter(
        red_points[:, 0],
        red_points[:, 1],
        color=plot_config["red_color"],
        s=plot_config["marker_size"],
        marker=".",
        zorder=3,
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
    axis.set_xticks(plot_config["x_ticks"])
    axis.set_yticks(plot_config["y_ticks"])
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )


def project_point_onto_line(point, anchor, direction):
    displacement = point - anchor
    return anchor + np.dot(displacement, direction) * direction


def choose_histogram_normal(
    features,
    anchor,
    direction,
):
    normal = np.array([-direction[1], direction[0]])

    if (
        np.dot(
            features.mean(axis=0) - anchor,
            normal,
        )
        < 0
    ):
        normal = -normal

    return normal


def draw_histogram_bar(
    axis,
    center,
    width,
    height,
    direction,
    normal,
    color,
    edge_color,
    line_width,
):
    half_width = 0.5 * width * direction

    vertices = np.vstack(
        (
            center - half_width,
            center + half_width,
            center + half_width + height * normal,
            center - half_width + height * normal,
        )
    )

    axis.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=line_width,
            zorder=2,
        )
    )


def draw_projection_histograms(
    axis,
    features,
    targets,
    class_mean_values,
    direction,
    anchor,
    config,
):
    plot_config = config["plot"]
    direction = normalize(direction)
    anchor = np.asarray(anchor, dtype=float)
    normal = choose_histogram_normal(
        features,
        anchor,
        direction,
    )

    projection_coordinates = (features - anchor) @ direction

    coordinate_min = projection_coordinates.min()
    coordinate_max = projection_coordinates.max()
    coordinate_range = coordinate_max - coordinate_min
    padding = plot_config["line_padding"] * coordinate_range

    line_start = anchor + (coordinate_min - padding) * direction
    line_end = anchor + (coordinate_max + padding) * direction

    axis.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color=plot_config["projection_line_color"],
        linestyle=plot_config["projection_line_style"],
        linewidth=plot_config["projection_line_width"],
        zorder=1,
    )

    bin_edges = np.linspace(
        coordinate_min,
        coordinate_max,
        plot_config["histogram_bins"] + 1,
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    colors = (
        plot_config["blue_color"],
        plot_config["red_color"],
    )

    histograms = []

    for class_label in (0, 1):
        class_coordinates = (features[targets == class_label] - anchor) @ direction

        counts, _ = np.histogram(
            class_coordinates,
            bins=bin_edges,
        )
        histograms.append(counts)

    maximum_count = max(counts.max() for counts in histograms)
    height_scale = plot_config["histogram_max_height"] / maximum_count

    for class_label, counts in enumerate(histograms):
        for center_coordinate, count in zip(
            bin_centers,
            counts,
        ):
            if count == 0:
                continue

            center = anchor + center_coordinate * direction

            draw_histogram_bar(
                axis=axis,
                center=center,
                width=bin_width * plot_config["histogram_width_fraction"],
                height=count * height_scale,
                direction=direction,
                normal=normal,
                color=colors[class_label],
                edge_color=plot_config["histogram_edge_color"],
                line_width=plot_config["histogram_edge_width"],
            )

    projected_means = np.vstack(
        [
            project_point_onto_line(
                mean,
                anchor,
                direction,
            )
            for mean in class_mean_values
        ]
    )

    for mean, projected_mean in zip(
        class_mean_values,
        projected_means,
    ):
        axis.plot(
            [mean[0], projected_mean[0]],
            [mean[1], projected_mean[1]],
            color=plot_config["mean_line_color"],
            linewidth=plot_config["mean_line_width"],
            zorder=4,
        )


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

    draw_projection_histograms(
        axis=axes[0],
        features=features,
        targets=targets,
        class_mean_values=means,
        direction=mean_direction_unit,
        anchor=config["plot"]["mean_projection_anchor"],
        config=config,
    )

    draw_projection_histograms(
        axis=axes[1],
        features=features,
        targets=targets,
        class_mean_values=means,
        direction=model.direction_,
        anchor=config["plot"]["fisher_projection_anchor"],
        config=config,
    )

    axes[0].set_title("Projection onto the line joining means")
    axes[1].set_title("Fisher linear discriminant")
    return figure


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
