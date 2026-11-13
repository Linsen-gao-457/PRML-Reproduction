import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, expit, ndtr

from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_figure(config):
    """Plot the threshold density and the resulting class probabilities."""
    plot = config["plot"]
    lower, upper = plot["score_range"]
    a = np.linspace(lower, upper, plot["grid_points"])
    a0 = plot["example_score"]
    theta = np.linspace(lower, a0, plot["grid_points"])
    normalizer = np.sqrt(2 * np.pi)
    probability = 0.5 * (1 + erf(a0 / np.sqrt(2)))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=plot["figure_size"], layout="constrained"
    )
    left.plot(a, np.exp(-(a**2) / 2) / normalizer, color="tab:blue")
    left.fill_between(
        theta, np.exp(-(theta**2) / 2) / normalizer, color="tab:blue", alpha=0.24
    )
    left.axvline(a0, linestyle=":", color="0.35")
    left.set(
        title=f"Gaussian threshold: P(class 1) = {probability:.3f}",
        xlabel=r"Threshold $\theta$",
        ylabel=r"Density $p(\theta)$",
        xlim=(lower, upper),
        ylim=(0, 0.44),
    )

    # Match the slopes of the probit and logistic curves at zero.
    probit = 0.5 * (1 + erf(a / np.sqrt(2)))
    logistic_scale = 4 / normalizer
    right.plot(a, probit, color="tab:blue", label=r"Probit $\Phi(a)$")
    right.plot(
        a,
        expit(logistic_scale * a),
        "--",
        color="tab:orange",
        label="Logistic (same slope at zero)",
    )
    right.scatter(a0, probability, color="tab:blue", zorder=4)
    right.set(
        title="Class 1 probability",
        xlabel=r"Score $a$",
        ylabel=r"$P(t=1\mid a)$",
        xlim=(lower, upper),
        ylim=(-0.02, 1.02),
    )
    right.legend(loc="lower right")
    for axis in (left, right):
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)

    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch04/probit_activation.yaml")
    figure = create_figure(config)
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close(figure)
    return output


def main():
    run()


if __name__ == "__main__":
    main()
