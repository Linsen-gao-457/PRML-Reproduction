import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import log_expit

from prml.classification import LaplaceApproximation
from prml.utils.config import load_config
from prml.utils.path import CONFIGS_DIR, PROJECT_ROOT
from prml.utils.plotting import save_figure


def create_figure(config):
    slope = config["target"]["slope"]
    bias = config["target"]["bias"]
    plot = config["plot"]

    def log_f(z):
        return -0.5 * z**2 + log_expit(slope * z + bias)

    def f(z):
        return np.exp(log_f(z))

    gaussian = LaplaceApproximation()(lambda z: f(z[0]), initial=[0.0])
    normalization, _ = quad(f, -np.inf, np.inf)
    z_min, z_max = plot["z_range"]
    z = np.linspace(z_min, z_max, plot["grid_points"])
    log_p = log_f(z) - np.log(normalization)
    log_q = gaussian.logpdf(z[:, None])

    figure, (left, right) = plt.subplots(
        1, 2, figsize=plot["figure_size"], sharex=True, layout="constrained"
    )
    left.fill_between(z, np.exp(log_p), color="#e4ce58", label=r"Target $p(z)$")
    left.plot(z, np.exp(log_q), color="red", label=r"Laplace $q(z)$")
    left.set(ylabel="Density", ylim=plot["density_limits"])

    right.plot(z, -log_p, color="#d6ba35", label=r"$-\log p(z)$")
    right.plot(z, -log_q, color="red", label=r"$-\log q(z)$")
    right.set(ylabel="Negative log density", ylim=plot["negative_log_limits"])

    for axis in (left, right):
        axis.set(xlabel=r"$z$", xlim=(z_min, z_max))
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend()
    return figure


def run():
    config = load_config(CONFIGS_DIR / "ch04/figure_4_14.yaml")

    figure = create_figure(config)

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
