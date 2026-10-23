import argparse

from .figure_4_4 import run as run_1
from .figure_4_5 import run as run_2

EXPERIMENTS = {
    "1": ("Running figure 4.4", run_1),
    "2": ("Running figure 4.5", run_2),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRML Chapter 4 experiments.")
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
        print(f"Running ch04 experiment {number}: {description}")
        run()


if __name__ == "__main__":
    main()
