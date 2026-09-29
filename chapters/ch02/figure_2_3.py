import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Bernoulli, Beta
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def likelihood_function(mu, observations):
    observations = np.asarray(observations, dtype=int)
    number_of_ones = np.count_nonzero(observations == 1)
    number_of_zeros = observations.size - number_of_ones
    return mu**number_of_ones * (1 - mu) ** number_of_zeros


def create_figure(prior, posterior, observations, grid_points, y_max):
    mu = np.linspace(start=0, stop=1, num=grid_points)
    prior_pdf = prior.pdf(mu)
    likelihood_func = likelihood_function(mu=mu, observations=observations)
    posterior_pdf = posterior.pdf(mu)
    figure, axes = plt.subplots(
        1,
        3,
        sharex=True,
        sharey=True,
    )
    data = [
        ("prior", prior_pdf, "red"),
        ("likelihood function", likelihood_func, "blue"),
        ("posterior", posterior_pdf, "red"),
    ]
    for axis, (label, values, color) in zip(axes.flat, data):
        axis.plot(
            mu,
            values,
            color=color,
            linewidth=2,
        )
        axis.text(
            0.05,
            0.85,
            label,
            transform=axis.transAxes,
        )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, y_max),
        xlabel=r"$\mu$",
    )
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_xticklabels(["0", "0.5", "1"])
    axis.set_yticks([0.0, 1.0, 2.0])
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    return figure


def run():
    config_path = CONFIGS_DIR / "ch02/figure_2_3.yaml"
    config = load_config(config_path)
    prior = Beta(a=config["prior"]["a"], b=config["prior"]["b"])
    observation = config["observation"]
    distribution = Bernoulli(mu=prior)
    distribution.fit(observations=observation)

    posterior = distribution.mu
    figure = create_figure(
        prior=prior,
        posterior=posterior,
        observations=observation,
        grid_points=config["grid_points"],
        y_max=config["y_max"],
    )
    output = save_figure(figure, PROJECT_ROOT / config["output"])
    plt.close()
    return output


def main():
    run()


if __name__ == "__main__":
    main()
