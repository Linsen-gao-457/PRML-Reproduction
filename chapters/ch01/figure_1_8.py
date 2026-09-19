import matplotlib.pyplot as plt
import numpy as np

from prml.linear_models import RidgeRegression, polynomial_basis
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure
from prml.utils.random import create_rng

from .figure_1_2 import generate_data


def rms(predictions, targets):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_errors(
    x_train,
    t_train,
    x_test,
    t_test,
    degree,
    log_regularization_values,
):
    train_design = polynomial_basis(x_train, degree)
    test_design = polynomial_basis(x_test, degree)

    train_errors = []
    test_errors = []

    for log_regularization in log_regularization_values:
        model = RidgeRegression(regularization=np.exp(log_regularization))
        model.fit(
            design_matrix=train_design,
            targets=t_train,
        )

        train_errors.append(rms(model.predict(train_design), t_train))
        test_errors.append(rms(model.predict(test_design), t_test))

    return np.asarray(train_errors), np.asarray(test_errors)


def create_figure(
    log_regularization_values,
    train_errors,
    test_errors,
):
    figure, axis = plt.subplots()

    axis.plot(
        log_regularization_values,
        train_errors,
        color="blue",
        label="Training",
    )
    axis.plot(
        log_regularization_values,
        test_errors,
        color="red",
        label="Test",
    )

    axis.set(
        xlabel=r"$\ln\lambda$",
        ylabel=r"$E_{\mathrm{RMS}}$",
        ylim=(0.0, 1.0),
        yticks=[0.0, 0.5, 1.0],
    )

    axis.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    return figure


def run(config_path=None):
    config_path = config_path or CONFIGS_DIR / "ch01/figure_1_8.yaml"
    config = load_config(config_path)
    rng = create_rng(config["seed"])

    train_size, test_size = config["sample_sizes"]

    x_train, t_train = generate_data(
        sample_size=train_size,
        noise_std=config["noise_std"],
        rng=rng,
    )
    x_test, t_test = generate_data(
        sample_size=test_size,
        noise_std=config["noise_std"],
        rng=rng,
    )

    log_min, log_max = config["log_regularization_range"]
    log_regularization_values = np.linspace(
        log_min,
        log_max,
        config["num_regularization_values"],
    )

    train_errors, test_errors = calculate_errors(
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        degree=config["degree"],
        log_regularization_values=log_regularization_values,
    )

    figure = create_figure(
        log_regularization_values,
        train_errors,
        test_errors,
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
