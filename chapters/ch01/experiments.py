from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .data import create_data, sin_function


def run_data_experiment(config):
    rng = np.random.default_rng(config["seed"])

    x, t = create_data(
        sample_size=config["sample_size"], noise_std=config["noise_std"], rng=rng
    )
    grid = np.linspace(0, 1, 200)
    fig, ax = plt.subplots()
    ax.plot(grid, sin_function(grid), color="green", label="sin function")
    ax.scatter(x, t, facecolors="none", edgecolors="blue", label="Training data")
    ax.set(xlabel="x", ylabel="t")
    ax.legend()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "Fig1-2.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path.resolve()}")
