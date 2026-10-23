import matplotlib.pyplot as plt
import numpy as np

from prml.classification import LeastSquaresClassifier, LogisticRegression
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def create_data(config, rng):
    data_config = config["data"]
    convariance = np.asarray(data_config["covariance"], dtype=float)
    red_points = rng.multivariate_normal(
        mean=data_config["red_mean"],
        cov=convariance,
        size=data_config["class_size"],
    )
    blue_points = rng.multivariate_normal(
        mean=data_config["blue_mean"],
        cov=convariance,
        size=data_config["class_size"],
    )
    outliers = rng.multivariate_normal(
        mean=data_config["outlier_mean"],
        cov=data_config["outlier_covariance"],
        size=data_config["outlier_size"],
    )
    features = np.vstack(
        (
            red_points,
            blue_points,
        )
    )
    targets = np.concatenate(
        (
            np.ones(red_points.shape[0], dtype=int),
            np.zeros(blue_points.shape[0], dtype=int),
        )
    )
    return features, targets, outliers


def least_squares_boundary(model, x):
    slope_difference = model.coef_[1] - model.coef_[0]
    intercept_difference = model.intercept_[1] - model.intercept_[0]
    return -(intercept_difference + slope_difference[0] * x) / slope_difference[1]


def logistic_regression_boundary(model, x):
    return -(model.intercept_ + model.coef_[0] * x) / model.coef_[1]


def draw_panel(axis, features, targets, least_squares, logistic_regression, config):
    plot_config = config["plot"]
    red_mask = targets == 1
    blue_mask = targets == 0
    axis.scatter(
        features[red_mask, 0],
        features[red_mask, 1],
        marker="x",
        color=plot_config["red_color"],
    )
    axis.scatter(
        features[blue_mask, 0],
        features[blue_mask, 1],
        marker="o",
        facecolors="none",
        edgecolors=plot_config["blue_color"],
    )
    x = np.linspace(
        plot_config["x_min"],
        plot_config["x_max"],
        plot_config["line_points"],
    )
    axis.plot(
        x,
        least_squares_boundary(
            least_squares,
            x,
        ),
        color=plot_config["least_squares_color"],
        label="least squares",
    )
    axis.plot(
        x,
        logistic_regression_boundary(
            logistic_regression,
            x,
        ),
        color=plot_config["logistic_regression_color"],
        label="logistic regression",
    )
    axis.set_xlim(
        plot_config["x_min"],
        plot_config["x_max"],
    )
    axis.set_ylim(
        plot_config["y_min"],
        plot_config["y_max"],
    )


def create_figure(config, rng):
    features, targets, outliers = create_data(config, rng)
    features_with_outliers = np.vstack((features, outliers))
    targets_with_outliers = np.concatenate(
        (targets, np.zeros(outliers.shape[0], dtype=int))
    )
    least_squares = LeastSquaresClassifier()
    logistic_regression = LogisticRegression()
    least_squares.fit(features, targets)
    logistic_regression.fit(features, targets)
    least_squares_with_outlier = LeastSquaresClassifier()
    logistic_regression_with_outlier = LogisticRegression()
    least_squares_with_outlier.fit(
        features_with_outliers,
        targets_with_outliers,
    )
    logistic_regression_with_outlier.fit(
        features_with_outliers,
        targets_with_outliers,
    )
    figrue, axes = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
    )
    draw_panel(
        axis=axes[0],
        features=features,
        targets=targets,
        least_squares=least_squares,
        logistic_regression=logistic_regression,
        config=config,
    )
    draw_panel(
        axis=axes[1],
        features=features_with_outliers,
        targets=targets_with_outliers,
        least_squares=least_squares_with_outlier,
        logistic_regression=logistic_regression_with_outlier,
        config=config,
    )
    axes[0].set_title("Original data")
    axes[1].set_title("With outliers")

    axes[0].set_xlabel(r"$x_1$")
    axes[1].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(
        r"$x_2$",
        rotation=0,
    )
    return figrue


def run():
    config = load_config(CONFIGS_DIR / "ch04/figure_4_4.yaml")

    rng = create_rng(seed=config["seed"])

    figure = create_figure(config, rng)

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
