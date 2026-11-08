"""Demonstrate Gaussian classification and sampling for three classes.

Run from the repository root with
``python -m chapters.ch04.gaussian_generative_classifier``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from prml.classification import GaussianGenerativeClassifier
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_data(config, n_samples, random_state):
    """Create labeled observations from shared-covariance Gaussians.

    Parameters
    ----------
    config : dict
        Data settings containing means, covariance, and class priors.
        Means must have two features for the demonstration plots.
    n_samples : int
        Number of observations to generate.
    random_state : int
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Generated input features.
    y : ndarray of shape (n_samples,)
        Integer class labels indexing the configured means.
    """
    rng = np.random.default_rng(random_state)
    means = np.asarray(config["means"], dtype=float)
    covariance = np.asarray(config["covariance"], dtype=float)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("The demo requires two features per class mean.")
    y = rng.choice(len(means), size=n_samples, p=config["priors"])
    X = means[y] + rng.multivariate_normal(np.zeros(2), covariance, n_samples)
    return X, y


def create_figure(model, X_test, y_test, X_generated, y_generated, config):
    """Plot decision regions, posterior confidence, and generated samples.

    Parameters
    ----------
    model : GaussianGenerativeClassifier
        Fitted two-feature classifier.
    X_test : ndarray of shape (n_test, 2)
        Held-out features.
    y_test : ndarray of shape (n_test,)
        Held-out class labels.
    X_generated : ndarray of shape (n_generated, 2)
        Features sampled from the fitted classifier.
    y_generated : ndarray of shape (n_generated,)
        Labels sampled from the fitted classifier.
    config : dict
        Plot settings loaded from YAML.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Three-panel demonstration figure.
    """
    points = np.vstack([X_test, X_generated, model.means_])
    lower = points.min(axis=0) - config["padding"]
    upper = points.max(axis=0) + config["padding"]
    xx, yy = np.meshgrid(
        np.linspace(lower[0], upper[0], config["grid_points"]),
        np.linspace(lower[1], upper[1], config["grid_points"]),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probabilities = model.predict_proba(grid)
    regions = probabilities.argmax(axis=1).reshape(xx.shape)

    colors = config["colors"]
    if len(colors) < model.k_classes_:
        raise ValueError("Configure at least one plot color per class.")
    figure, axes = plt.subplots(
        1, 3, figsize=config["figsize"], sharex=True, sharey=True
    )
    axes[0].contourf(
        xx,
        yy,
        regions,
        levels=np.arange(model.k_classes_ + 1) - 0.5,
        cmap=ListedColormap(colors[:model.k_classes_]),
        alpha=config["region_alpha"],
    )
    confidence = axes[1].contourf(
        xx,
        yy,
        probabilities.max(axis=1).reshape(xx.shape),
        levels=np.linspace(1 / model.k_classes_, 1, config["confidence_levels"]),
        cmap=config["confidence_cmap"],
        vmin=1 / model.k_classes_,
        vmax=1,
    )
    figure.colorbar(confidence, ax=axes[1], label="Maximum posterior probability")

    for k, label in enumerate(model.classes_):
        selected = y_test == label
        axes[0].scatter(
            X_test[selected, 0],
            X_test[selected, 1],
            color=colors[k],
            s=config["marker_size"],
            label=f"Class {label}",
        )
        selected = y_generated == label
        axes[2].scatter(
            X_generated[selected, 0],
            X_generated[selected, 1],
            color=colors[k],
            s=config["marker_size"],
            alpha=config["generated_alpha"],
            label=f"Class {label}",
        )

    accuracy = np.mean(model.predict(X_test) == y_test)
    titles = [
        f"Held-out classification (accuracy: {accuracy:.1%})",
        "Posterior confidence",
        "Samples from the fitted model",
    ]
    for axis, title in zip(axes, titles):
        axis.scatter(
            model.means_[:, 0],
            model.means_[:, 1],
            marker="X",
            c="black",
            s=config["mean_marker_size"],
            label="Learned means",
            zorder=4,
        )
        axis.set(
            title=title,
            xlabel="$x_1$",
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
        )
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(fontsize="small")
    axes[2].legend(fontsize="small")
    figure.tight_layout()
    return figure


def run():
    """Fit, evaluate, sample, and save the demonstration figure.

    Returns
    -------
    output : pathlib.Path
        Saved figure path.
    """
    config = load_config(CONFIGS_DIR / "ch04/gaussian_generative_classifier.yaml")
    data = config["data"]
    X_train, y_train = create_data(data, data["train_samples"], data["train_seed"])
    X_test, y_test = create_data(data, data["test_samples"], data["test_seed"])
    model = GaussianGenerativeClassifier(reg=config["model"]["reg"]).fit(
        X_train, y_train
    )

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    X_generated, y_generated = model.sample(
        n_samples=config["sampling"]["n_samples"],
        random_state=config["sampling"]["seed"],
    )

    # print(f"Classes: {model.classes_}")
    # print(f"k_classes_: {model.k_classes_}, m_features_: {model.m_features_}")
    # print(f"Learned priors: {model.priors_}")
    # print(f"Held-out accuracy: {np.mean(predictions == y_test):.3f}")
    # print(f"First five posterior probability rows:\n{probabilities[:5]}")
    # print(f"Generated shapes: X={X_generated.shape}, y={y_generated.shape}")

    figure = create_figure(
        model, X_test, y_test, X_generated, y_generated, config["plot"]
    )
    output = save_figure(
        figure, PROJECT_ROOT / config["output"], dpi=config["plot"]["dpi"]
    )
    plt.close(figure)
    return output


if __name__ == "__main__":
    run()
