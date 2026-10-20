import argparse


EXPERIMENTS = {
    # "1": ("Running figure 3.1", run_1),
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
        print(f"Running ch03 experiment {number}: {description}")
        run()


if __name__ == "__main__":
    main()
