import argparse

from .figure_2_2 import run as run_figure_2_2

# from .figure_1_4 import run as run_figure_1_4
# from .figure_1_5 import run as run_figure_1_5
# from .figure_1_6 import run as run_figure_1_6
# from .figure_1_7 import run as run_figure_1_7
# from .figure_1_8 import run as run_figure_1_8

EXPERIMENTS = {
    "1": ("Running figure 2.2", run_figure_2_2),
    # "2": ("Running figure 1.4", run_figure_1_4),
    # "3": ("Running figure 1.5", run_figure_1_5),
    # "4": ("Running figure 1.6", run_figure_1_6),
    # "5": ("Running figure 1.7", run_figure_1_7),
    # "6": ("Running figure 1.8", run_figure_1_8),
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
        print(f"Running ch01 experiment {number}: {description}")
        run()


if __name__ == "__main__":
    main()
