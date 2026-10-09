import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Gaussian
from prml.distributions.student_t import Student_T
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def fit_distributions(
    observations,
    learning_rate,
    max_iterations,
    tolerance,
):
    gaussian = Gaussian()
    gaussian.fit(observations)

    student_t = Student_T()
    student_t.fit(
        observations,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    return gaussian, student_t


def plot_panel(
    axis,
    x,
    observations,
    gaussian,
    student_t,
    bins,
    x_min,
    x_max,
    y_min,
    y_max,
    label,
):
    axis.hist(
        observations,
        bins=bins,
        range=(x_min, x_max),
        density=True,
        color="#7777cc",
        edgecolor="black",
        linewidth=0.8,
    )

    axis.plot(
        x,
        gaussian.pdf(x),
        color="green",
        linewidth=2,
        label="Gaussian",
    )

    axis.plot(
        x,
        student_t.pdf(x),
        color="red",
        linewidth=2,
        label="Student-t",
    )

    axis.set(
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
    )

    axis.set_xticks([-5.0, 0.0, 5.0, 10.0])
    axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    axis.text(
        0.5,
        -0.1,
        label,
        transform=axis.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
    )


def create_figure(
    clean_observations,
    contaminated_observations,
    learning_rate,
    max_iterations,
    tolerance,
    x_min,
    x_max,
    y_min,
    y_max,
    grid_points,
    bins,
):
    x = np.linspace(
        start=x_min,
        stop=x_max,
        num=grid_points,
    )

    clean_gaussian, clean_student_t = fit_distributions(
        observations=clean_observations,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    contaminated_gaussian, contaminated_student_t = fit_distributions(
        observations=contaminated_observations,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        sharex=True,
        sharey=True,
    )

    plot_panel(
        axis=axes[0],
        x=x,
        observations=clean_observations,
        gaussian=clean_gaussian,
        student_t=clean_student_t,
        bins=bins,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        label="(a)",
    )

    plot_panel(
        axis=axes[1],
        x=x,
        observations=contaminated_observations,
        gaussian=contaminated_gaussian,
        student_t=contaminated_student_t,
        bins=bins,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        label="(b)",
    )

    axes[1].legend(
        frameon=False,
        loc="upper right",
    )

    figure.tight_layout()
    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_16.yaml"
    config = load_config(config_path)

    rng = create_rng(seed=config["seed"])

    clean_observations = rng.normal(
        loc=config["data"]["mean"],
        scale=config["data"]["standard_deviation"],
        size=config["data"]["sample_size"],
    )

    contaminated_observations = np.concatenate(
        (
            clean_observations,
            np.asarray(
                config["data"]["outliers"],
                dtype=float,
            ),
        )
    )

    figure = create_figure(
        clean_observations=clean_observations,
        contaminated_observations=(contaminated_observations),
        learning_rate=config["student_t"]["learning_rate"],
        max_iterations=config["student_t"]["max_iterations"],
        tolerance=config["student_t"]["tolerance"],
        x_min=config["plot"]["x_min"],
        x_max=config["plot"]["x_max"],
        y_min=config["plot"]["y_min"],
        y_max=config["plot"]["y_max"],
        grid_points=config["plot"]["grid_points"],
        bins=config["plot"]["bins"],
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
