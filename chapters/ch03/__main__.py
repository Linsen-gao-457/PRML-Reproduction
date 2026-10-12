import argparse

from .figure_3_1 import run as run_1
from .figure_3_2_plus import run as run_2
from .figure_3_5 import run as run_5
from .ridge_regression import run as run_4
from .sequential_learning import run as run_3

EXPERIMENTS = {
    "1": ("Running figure 3.1", run_1),
    "2": ("Running figure 3.2 plus", run_2),
    "3": ("Running sequential learning", run_3),
    "4": ("Running ridge regression", run_4),
    "5": ("Running figure 3.5", run_5),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRML Chapter 3 experiments.")
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
        print(f"Running ch03 experiment {number}: {description}")
        run()


if __name__ == "__main__":
    main()
