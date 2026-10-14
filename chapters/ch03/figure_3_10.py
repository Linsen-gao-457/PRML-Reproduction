import matplotlib.pyplot as plt
import numpy as np

from prml.distributions.multivariate_gaussian import MultivariateGaussian
from prml.linear_models import BayesianLinearRegression, gaussian_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_posterior_covariance(design_matrix, alpha, precision):
    number_of_features = design_matrix.shape[1]

    prior = MultivariateGaussian(
        mean=np.zeros(number_of_features),
        cov=np.eye(number_of_features) / alpha,
    )

    model = BayesianLinearRegression(
        prior=prior,
        precision=precision,
    )

    for basis in design_matrix:
        model.update(
            basis=basis,
            target=0.0,
        )

    return model.posterior_covariance


def equivalent_kernel(
    design_matrix,
    posterior_covariance,
    precision,
):
    return precision * np.einsum(
        "im,mn,jn->ij",
        design_matrix,
        posterior_covariance,
        design_matrix,
    )


def nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def create_figure(config):
    data_config = config["data"]
    basis_config = config["basis"]
    model_config = config["model"]
    plot_config = config["plot"]
    x = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        data_config["number_of_points"],
    )

    centers = np.linspace(
        data_config["x_min"],
        data_config["x_max"],
        basis_config["number_of_centers"],
    )
    design_matrix = gaussian_basis(x, centers, basis_config["width"])
    precision = 1.0 / data_config["noise_std"] ** 2

    posterior_covariance = create_posterior_covariance(
        design_matrix=design_matrix,
        alpha=model_config["alpha"],
        precision=precision,
    )
    kernel = equivalent_kernel(
        design_matrix=design_matrix,
        posterior_covariance=posterior_covariance,
        precision=precision,
    )
    figure = plt.figure(
        figsize=plot_config["figure_size"],
    )
    grid = figure.add_gridspec(
        nrows=len(plot_config["slice_locations"]),
        ncols=2,
        width_ratios=plot_config["width_ratios"],
        wspace=plot_config["column_spacing"],
        hspace=plot_config["row_spacing"],
    )

    kernel_axis = figure.add_subplot(grid[:, 1])
    kernel_axis.imshow(
        kernel,
        origin="lower",
        extent=[
            data_config["x_min"],
            data_config["x_max"],
            data_config["x_min"],
            data_config["x_max"],
        ],
        cmap=plot_config["color_map"],
        interpolation=plot_config["interpolation"],
        aspect="equal",
    )

    kernel_axis.set_xlabel(r"$x'$")
    kernel_axis.set_ylabel(r"$x$", rotation=0)
    kernel_axis.set_xticks([-1, 0, 1])
    kernel_axis.set_yticks([-1, 0, 1])
    kernel_axis.tick_params(
        direction="in",
        top=True,
        right=True,
    )

    for row, slice_location in enumerate(plot_config["slice_locations"]):
        slice_axis = figure.add_subplot(grid[row, 0])
        slice_index = nearest_index(x, slice_location)
        kernel_slice = kernel[slice_index]

        slice_axis.plot(
            x,
            kernel_slice,
            color=plot_config["slice_color"],
            linewidth=plot_config["slice_linewidth"],
        )

        slice_axis.axhline(
            0.0,
            color=plot_config["guide_color"],
            linewidth=plot_config["guide_linewidth"],
        )

        kernel_axis.axhline(
            x[slice_index],
            color=plot_config["guide_color"],
            linewidth=plot_config["guide_linewidth"],
        )

        slice_axis.set_xlim(
            data_config["x_min"],
            data_config["x_max"],
        )

        slice_axis.set_ylim(
            plot_config["slice_y_min"],
            plot_config["slice_y_max"],
        )

        slice_axis.set_xticks([])
        slice_axis.set_yticks([])

        for spine in slice_axis.spines.values():
            spine.set_visible(False)

    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch03/figure_3_10.yaml")

    figure = create_figure(config=config)

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
