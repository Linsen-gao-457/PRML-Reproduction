import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import MultivariateGaussian
from prml.linear_models import BayesianLinearRegression, gaussian_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng


def sinusodial(x):
    pass


def create_data(config, rng):
    pass


def draw_panel():
    pass


def create_figure(config, rng):
    data_config = config["data"]
    model_config = config["model"]
    plot_config = config["plot"]
    x_train, t_train = create_data(config, rng)
    x_plot = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        plot_config["grid_points"],
    )
    centers = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        model_config["number_of_basis_functions"],
    )
    train_matrix = gaussian_basis(
        x_train,
        centers=centers,
        width=model_config["basis_width"],
    )

    plotting_matrix = gaussian_basis(
        x=x_plot,
        centers=centers,
        width=model_config["basis_width"],
    )

    prior = MultivariateGaussian(
        mean=np.zeros(model_config["number_of_basis_functions"]),
        cov=(np.eye(model_config["number_of_basis_functions"]) / model_config["alpha"]),
    )
    model = BayesianLinearRegression(
        prior=prior,
        precision=1.0 / data_config["noise_std"] ** 2,
    )
    figure, axes = plt.subplots(
        2,
        2,
        sharex=True,
        sharey=True,
    )
    sample_sizes = data_config["sample_sizes"]
    axes_by_sample_size = dict(zip(sample_sizes, axes.flat))
    for sample_size, (basis, target) in enumerate(
        zip(train_matrix, t_train),
        start=1,
    ):
        model.update(
            basis=basis,
            target=target,
        )
    for sample_size, (basis, target) in enumerate(
        zip(train_matrix, t_train),
        start=1,
    ):
        model.update(
            basis=basis,
            target=target,
        )

        if sample_size not in axes_by_sample_size:
            continue

        sampled_weights = model.sample_weight(
            sample_size=model_config["number_of_weight_samples"],
            rng=rng,
        )
        sample_functions = np.einsum(
            "ij, kj -> ik",
            plotting_matrix,
            sampled_weights,
        )
        draw_panel()

    figure.tight_layout()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_3_9.yaml")

    rng = create_rng(seed=config["seed"])

    figure = create_figure(config=config, rng=rng)

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
