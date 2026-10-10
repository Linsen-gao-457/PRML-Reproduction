import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0e

from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def von_mises_pdf(theta, mean_direction, concentration):
    """Evaluate the von Mises density using a stable normalization."""
    log_i0 = np.log(i0e(concentration)) + concentration
    log_density = (
        concentration * np.cos(theta - mean_direction)
        - np.log(2.0 * np.pi)
        - log_i0
    )
    return np.exp(log_density)


def plot_direction(axis, angle, length, label, label_offset=(0.0, 0.0)):
    endpoint_x = length * np.cos(angle)
    endpoint_y = length * np.sin(angle)
    axis.plot(
        [0.0, endpoint_x],
        [0.0, endpoint_y],
        color="black",
        linewidth=1.5,
    )
    axis.text(
        endpoint_x + label_offset[0],
        endpoint_y + label_offset[1],
        label,
        fontsize=13,
        horizontalalignment="center",
        verticalalignment="center",
    )


def create_figure(curves, plot_config):
    grid_points = plot_config["grid_points"]
    theta = np.linspace(0.0, 2.0 * np.pi, grid_points)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(
            plot_config["figure_width"],
            plot_config["figure_height"],
        ),
    )
    cartesian_axis, circular_axis = axes

    for curve in curves:
        density = von_mises_pdf(
            theta=theta,
            mean_direction=curve["mean_direction"],
            concentration=curve["concentration"],
        )

        cartesian_axis.plot(
            theta,
            density,
            color=curve["color"],
            linewidth=2,
            label=curve["label"],
        )

        circular_axis.plot(
            density * np.cos(theta),
            density * np.sin(theta),
            color=curve["color"],
            linewidth=2,
            label=curve["label"],
        )

    cartesian_axis.set(
        xlim=(0.0, 2.0 * np.pi),
        ylim=(0.0, plot_config["density_y_max"]),
        xlabel=r"$\theta$",
        ylabel=r"$p(\theta)$",
    )
    cartesian_axis.set_xticks(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi],
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    cartesian_axis.tick_params(direction="in", top=True, right=True)
    cartesian_axis.legend(frameon=False, loc="upper right")

    ray_length = plot_config["ray_length"]
    plot_direction(
        circular_axis,
        angle=0.0,
        length=ray_length,
        label=r"$0$",
        label_offset=(0.0, 0.06),
    )
    circular_axis.text(
        ray_length,
        -0.08,
        r"$2\pi$",
        fontsize=13,
        horizontalalignment="center",
    )
    plot_direction(
        circular_axis,
        angle=np.pi / 4.0,
        length=ray_length,
        label=r"$\pi/4$",
        label_offset=(0.02, 0.08),
    )
    plot_direction(
        circular_axis,
        angle=3.0 * np.pi / 4.0,
        length=plot_config["three_quarter_ray_length"],
        label=r"$3\pi/4$",
        label_offset=(-0.08, 0.08),
    )

    circular_axis.scatter([0.0], [0.0], color="black", s=12, zorder=3)
    circular_axis.set(
        xlim=(
            plot_config["circular_x_min"],
            plot_config["circular_x_max"],
        ),
        ylim=(
            plot_config["circular_y_min"],
            plot_config["circular_y_max"],
        ),
        aspect="equal",
    )
    circular_axis.set_xticks([])
    circular_axis.set_yticks([])
    circular_axis.legend(frameon=False, loc="lower left")

    figure.tight_layout()
    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_19.yaml"
    config = load_config(config_path)
    figure = create_figure(
        curves=config["curves"],
        plot_config=config["plot"],
    )
    output = save_figure(
        figure,
        PROJECT_ROOT / config["output"],
    )
    plt.close(figure)
    return output


def main():
    output = run()
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
