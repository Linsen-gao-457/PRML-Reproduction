import argparse

from .bernoulli_frequentisist_bayesian import run as run_2
from .categorical_dirichlet_bernoulli import run as run_4
from .figure_2_2 import run as run_1
from .figure_2_3 import run as run_3
from .figure_2_6 import run as run_5
from .figure_2_12 import run as run_6
from .figure_2_13_plus import run as run_7
from .figure_2_14_plus import run as run_8
from .figure_2_15 import run as run_9

EXPERIMENTS = {
    "1": ("Running figure 2.2", run_1),
    "2": ("Running bayesian and frequentist on Bernoulli distribution", run_2),
    "3": ("Running figure 2.3", run_3),
    "4": ("categorical_dirichlet_bernoulli", run_4),
    "5": ("Running figure 2.6", run_5),
    "6": ("Running figure 2.12", run_6),
    "7": ("Running figure 2.13_plus", run_7),
    "8": ("Running figure 2.14_plus", run_8),
    "9": ("Running figure 2.15", run_9),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PRML Chapter 2 experiments.")
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
        print(f"Running ch02 experiment {number}: {description}")
        run()


if __name__ == "__main__":
    main()
