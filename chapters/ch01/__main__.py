import argparse

from .figure_1_2 import run as run_figure_1_2
from .figure_1_4 import run as run_figure_1_4

EXPERIMENTS = {
    "1": ("Running figure 1.2", run_figure_1_2),
    "2": ("Running figure 1.4", run_figure_1_4),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRML Chapter 1 experiments.")
    parser.add_argument(
        "experiment",
        choices=[*EXPERIMENTS, "all"],
        help="experiment number, or 'all' to run every implemented experiment",
    )
    return parser.parse_args()


def main():
    selected = parse_args().experiment
    experiment_numbers = EXPERIMENTS if selected == "all" else [selected]

    for number in experiment_numbers:
        description, run = EXPERIMENTS[number]
        print(f"Running ch01 experiment {number}: {description}")
        run()


if __name__ == "__main__":
    main()
