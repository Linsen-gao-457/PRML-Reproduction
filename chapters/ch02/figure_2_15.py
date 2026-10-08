import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Gaussian
from prml.distributions.student_t import Student_T
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_figure(
    mean,
    precision,
    student_t_curves,
    gaussian_limit,
    x_min,
    x_max,
    y_min,
    y_max,
    grid_points,
):
    x = np.linspace(
        start=x_min,
        stop=x_max,
        num=grid_points,
    )

    figure, axis = plt.subplots()

    for curve in student_t_curves:
        distribution = Student_T(
            mean=mean,
            precision=precision,
            degree_of_freedom=curve["degree_of_freedom"],
        )

        axis.plot(
            x,
            distribution.pdf(x),
            color=curve["color"],
            linewidth=2,
            label=curve["label"],
        )

    # As nu approaches infinity, Student-t approaches
    # a Gaussian with variance 1 / precision.
    gaussian = Gaussian(
        mean=mean,
        standard_deviation=np.sqrt(1.0 / precision),
    )

    axis.plot(
        x,
        gaussian.pdf(x),
        color=gaussian_limit["color"],
        linewidth=2,
        label=gaussian_limit["label"],
    )

    axis.set(
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        xlabel=r"$x$",
        ylabel=r"$p(x)$",
    )

    axis.set_xticks([-5.0, 0.0, 5.0])
    axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    axis.legend(
        frameon=False,
        loc="upper right",
    )

    figure.tight_layout()
    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_15.yaml"
    config = load_config(config_path)

    figure = create_figure(
        mean=config["mean"],
        precision=config["precision"],
        student_t_curves=config["student_t_curves"],
        gaussian_limit=config["gaussian_limit"],
        x_min=config["plot"]["x_min"],
        x_max=config["plot"]["x_max"],
        y_min=config["plot"]["y_min"],
        y_max=config["plot"]["y_max"],
        grid_points=config["plot"]["grid_points"],
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
