import matplotlib.pyplot as plt
import numpy as np

from prml.distributions import Bernoulli, Categorical, Dirichlet
from prml.utils.path import PROJECT_ROOT
from prml.utils.plotting import save_figure

OUTPUT_PATH = PROJECT_ROOT / "outputs/figures/categorical_dirichlet_bernoulli.png"


def categorical_dirichlet_example():
    """Fit a categorical model using MLE and Bayesian inference."""
    observations = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )

    # Frequentist estimate
    mle_model = Categorical(
        mu=np.full(observations.shape[1], 1.0 / observations.shape[1])
    ).fit(observations)

    # Bayesian estimate with a Dirichlet prior
    prior = Dirichlet(alpha=np.array([2.0, 2.0, 2.0]))
    bayesian_model = Categorical(mu=prior).fit(observations)
    posterior = bayesian_model.mu

    np.testing.assert_allclose(prior.alpha, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(posterior.alpha, [5.0, 4.0, 3.0])

    return mle_model, posterior


def bernoulli_is_binary_categorical():
    """
    Bernoulli(mu=p) is equivalent to a two-class categorical
    distribution with probabilities [1-p, p].
    """
    probability_of_one = 0.7

    bernoulli = Bernoulli(mu=probability_of_one)
    categorical = Categorical(
        mu=np.array([1.0 - probability_of_one, probability_of_one])
    )

    binary_values = np.array([0, 1])
    one_hot_values = np.eye(2, dtype=int)[binary_values]

    bernoulli_probabilities = bernoulli.pmf(binary_values)
    categorical_probabilities = categorical.pmf(one_hot_values)

    np.testing.assert_allclose(
        bernoulli_probabilities,
        categorical_probabilities,
    )

    return bernoulli_probabilities, categorical_probabilities


def create_figure(
    mle_model,
    posterior,
    bernoulli_probabilities,
    categorical_probabilities,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Categorical and Dirichlet relationship
    classes = np.arange(3)
    width = 0.25

    axes[0].bar(
        classes - width,
        np.full(3, 1.0 / 3.0),
        width,
        label="Dirichlet prior mean",
        color="tab:gray",
    )
    axes[0].bar(
        classes,
        mle_model.mean,
        width,
        label="Categorical MLE",
        color="tab:blue",
    )
    axes[0].bar(
        classes + width,
        posterior.mean,
        width,
        label="Posterior mean",
        color="tab:orange",
    )

    axes[0].set(
        title="Categorical likelihood with Dirichlet prior",
        xlabel="Class",
        ylabel="Probability",
        xticks=classes,
        xticklabels=["Class 0", "Class 1", "Class 2"],
        ylim=(0, 0.7),
    )
    axes[0].legend()

    # Bernoulli as a two-class categorical distribution
    binary_values = np.array([0, 1])

    axes[1].bar(
        binary_values - 0.15,
        bernoulli_probabilities,
        width=0.3,
        label="Bernoulli",
        color="tab:green",
    )
    axes[1].bar(
        binary_values + 0.15,
        categorical_probabilities,
        width=0.3,
        label="Categorical",
        color="tab:red",
        alpha=0.75,
    )

    axes[1].set(
        title="Bernoulli is a binary categorical",
        xlabel="Outcome",
        ylabel="Probability",
        xticks=binary_values,
        xticklabels=["0", "1"],
        ylim=(0, 1),
    )
    axes[1].legend()

    fig.tight_layout()
    return fig


def run():
    mle_model, posterior = categorical_dirichlet_example()

    bernoulli_probabilities, categorical_probabilities = (
        bernoulli_is_binary_categorical()
    )

    print("Categorical MLE:", mle_model.mean)
    print("Dirichlet posterior alpha:", posterior.alpha)
    print("Dirichlet posterior mean:", posterior.mean)
    print("Bernoulli PMF:", bernoulli_probabilities)
    print("Categorical PMF:", categorical_probabilities)
    print("Bernoulli and binary categorical PMFs are equal.")

    figure = create_figure(
        mle_model,
        posterior,
        bernoulli_probabilities,
        categorical_probabilities,
    )

    save_figure(figure, OUTPUT_PATH, dpi=200)
    plt.close(figure)


def main():
    run()


if __name__ == "__main__":
    main()
