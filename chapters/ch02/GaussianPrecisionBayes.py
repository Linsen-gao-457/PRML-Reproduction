import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Gamma, GaussianPrecisionBayes
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


COLORS = {
    0: "black",
    1: "green",
    2: "blue",
    10: "red",
}


def create_figure(
    observations,
    known_mean,
    prior,
    sample_sizes,
    grid_points,
    x_min,
    x_max,
    y_min,
    y_max,
    true_precision,
):
    precision = np.linspace(
        start=x_min,
        stop=x_max,
        num=grid_points,
    )

    figure, axis = plt.subplots()

    axis.plot(
        precision,
        prior.pdf(precision),
        color=COLORS[0],
        linewidth=2,
        label=r"$N=0$",
    )

    for sample_size in sample_sizes:
        model = GaussianPrecisionBayes(
            mean=known_mean,
            prior=prior,
        )

        posterior = model.fit(observations[:sample_size])

        axis.plot(
            precision,
            posterior.pdf(precision),
            color=COLORS[sample_size],
            linewidth=2,
            label=rf"$N={sample_size}$",
        )

    # Mark the true precision used to generate the observations.
    axis.axvline(
        true_precision,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=rf"true $\lambda={true_precision:g}$",
    )

    axis.set(
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        xlabel=r"precision $\lambda$",
        ylabel=r"$p(\lambda \mid \mathcal{D})$",
    )

    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    axis.legend(frameon=False)
    figure.tight_layout()

    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_gaussian_precision_bayes.yaml"
    config = load_config(config_path)

    rng = create_rng(seed=config["seed"])

    data_mean = config["data"]["mean"]
    data_variance = config["data"]["variance"]

    observations = rng.normal(
        loc=data_mean,
        scale=np.sqrt(data_variance),
        size=config["data"]["data_sample_size"],
    )

    prior = Gamma(
        a=config["prior"]["a"],
        b=config["prior"]["b"],
    )

    figure = create_figure(
        observations=observations,
        known_mean=data_mean,
        prior=prior,
        sample_sizes=config["posterior_sample_sizes"],
        grid_points=config["plot"]["grid_points"],
        x_min=config["plot"]["x_min"],
        x_max=config["plot"]["x_max"],
        y_min=config["plot"]["y_min"],
        y_max=config["plot"]["y_max"],
        true_precision=1.0 / data_variance,
    )

    output = save_figure(
        figure,
        PROJECT_ROOT / config["output"],
    )

    plt.close(figure)
    return output


def main():
    output = run()
    return output


if __name__ == "__main__":
    main()
