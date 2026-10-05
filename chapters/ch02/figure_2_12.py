import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Gaussian, GaussianMeanBayes
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
    sample_dataset,
    prior,
    precision,
    sample_sizes,
    grid_points,
    x_min,
    x_max,
    y_min,
    y_max,
):
    x = np.linspace(start=x_min, stop=x_max, num=grid_points)
    figure, axis = plt.subplots()
    axis.plot(x, prior.pdf(x), color=COLORS[0], label="$N=0$")
    for sample_size in sample_sizes:
        model = GaussianMeanBayes(prior=prior, precision=precision)
        posterior = model.fit(sample_dataset[:sample_size])
        axis.plot(
            x,
            posterior.pdf(x),
            color=COLORS[sample_size],
            label=rf"$N={sample_size}$",
        )

    axis.set(
        xlim=(x_min, x_max),
        ylim=(0.0, y_max),
        xlabel=r"$\mu$",
        ylabel=r"$p(\mu \mid \mathcal{D})$",
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
    configure_path = CONFIGS_DIR / "ch02/figure_2_12.yaml"
    config = load_config(configure_path)
    rng = create_rng(seed=config["seed"])
    sample_dataset = rng.normal(
        loc=config["data"]["mean"],
        scale=np.sqrt(config["data"]["variance"]),
        size=config["data"]["data_sample_size"],
    )
    prior = Gaussian(
        mean=config["prior"]["mean"],
        standard_deviation=np.sqrt(config["prior"]["variance"]),
    )
    figure = create_figure(
        sample_dataset=sample_dataset,
        prior=prior,
        precision=1 / config["data"]["variance"],
        sample_sizes=config["posterior_sample_sizes"],
        grid_points=config["plot"]["grid_points"],
        x_min=config["plot"]["x_min"],
        x_max=config["plot"]["x_max"],
        y_min=config["plot"]["y_min"],
        y_max=config["plot"]["y_max"],
    )
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close(figure)
    return output


def main():
    run()


if __name__ == "__main__":
    main()
