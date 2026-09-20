import numpy as np
from prml.distributions import Bernoulli, Beta


def frequentist_ber_example():
    observations = np.array([1, 0, 1, 1, 0])
    distribution = Bernoulli().fit(observations)
    print("Frequentist estimate")
    print(f"Mean: {distribution.mean}")
    print(f"Variance: {distribution.variance}")
    print(f"PMF for [0, 1]: {distribution.pmf([0, 1])}")
    rng = np.random.default_rng(1234)
    samples = distribution.draw(10, rng=rng)
    print(f"Samples: {samples}")


def bayesian_ber_example():
    prior_distribution = Beta(a=2, b=2)
    distribution = Bernoulli(mu=prior_distribution)
    observations = np.array([1, 0, 1, 1, 0])
    distribution.fit(observations)
    posterior_distribution = distribution.mu
    print("Bayesian estimate")
    print(f"Posterior a: {posterior_distribution.a}")
    print(f"Posterior b: {posterior_distribution.b}")
    print(f"Posterior mean: {posterior_distribution.mean}")


def run():
    frequentist_ber_example()
    print("----------------")
    bayesian_ber_example()


def main():
    run()


if __name__ == "__main__":
    main()
