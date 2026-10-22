import matplotlib.pyplot as plt
import numpy as np
from prml.classification import LogisticRegression, LeastSquaresClassifier
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


def draw_panel(axis, features, targets, least_squares, logistic_regression, config):
    pass


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
