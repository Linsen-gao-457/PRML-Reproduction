import argparse
import json
from pathlib import Path

from .experiments import run_data_experiment


def main():
    experiments = {
        "1": run_data_experiment,
        # "2": run_polynomial_experiment,
    }
    parser = argparse.ArgumentParser(description="Run PRML Chapter 1 experiments.")
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        choices=["all", *experiments],
        help="1: reproduce Fig 1.2, all: run all experiments in ch01.",
    )
    args = parser.parse_args()
    config_path = Path(__file__).with_name("config.json")

    with config_path.open() as file:
        config = json.load(file)

    # Command argument overriders the configuration.
    selected = (
        args.experiment
        if args.experiment is None
        else str(config.get("experiment", "all"))
    )

    if selected != "all" and selected not in experiments:
        parser.error(f"Unknown configured experiment: {selected!r}")
    if selected == "all":
        for number, function in experiments.items():
            print(f"Running experiment {number}")
            function(config)
    else:
        print(f"Running experiment {selected}")
        experiments[selected](config)


if __name__ == "__main__":
    main()
