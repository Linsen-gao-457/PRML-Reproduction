import argparse


EXPERIMENTS = {
    # "1": ("Running figure 2.2", run_1),
    # "2": ("Running bayesian and frequentist on Bernoulli distribution", run_2),
    # "3": ("Running figure 2.3", run_3),
    # "4": ("categorical_dirichlet_bernoulli", run_4),
    # "5": ("Running figure 2.6", run_5),
    # "6": ("Running figure 2.12", run_6),
    # "7": ("Running figure 2.13_plus", run_7),
    # "8": ("Running figure 2.14_plus", run_8),
    # "9": ("Running figure 2.15", run_9),
    # "10": ("Running figure 2.16", run_10),
    # "11": ("Running figure 2.19", run_11),
    # "12": ("Running figure 2.24", run_12),
    # "13": ("Running figure 2.25", run_13),
    # "14": ("Running figure 2.26", run_14),
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
