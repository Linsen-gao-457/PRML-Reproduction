# Chapter 2: Probability Distributions

Reproductions and demonstrations of the probability distributions introduced
in Chapter 2 of Christopher M. Bishop's _Pattern Recognition and Machine
Learning_ (PRML).

## Running the experiments

Install the project from the repository root, then run a Chapter 2 experiment
by number. For example:

```bash
python -m chapters.ch02 1
```

Run every implemented Chapter 2 experiment with:

```bash
python -m chapters.ch02 all
```

Figure experiments read their parameters from `configs/ch02/` and save plots
under `outputs/figures/`.

## Implemented experiments

| Experiment | Figure or demonstration | Concept                                                                                           | Configuration                  | Output                                                |
| ---------: | :---------------------: | ------------------------------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------- |
|        `1` |       Figure 2.2        | Beta densities for several values of $a$ and $b$                                                  | `configs/ch02/figure_2_2.yaml` | `outputs/figures/figure_2_2.png`                      |
|        `2` |     Bernoulli demo      | Frequentist estimation and beta-Bernoulli Bayesian updating                                       | —                              | Terminal output                                       |
|        `3` |       Figure 2.3        | One step of sequential Bayesian inference for Bernoulli data                                      | `configs/ch02/figure_2_3.yaml` | `outputs/figures/figure_2_3.png`                      |
|        `4` |    Categorical demo     | Categorical MLE, Dirichlet posterior updating, and Bernoulli as a binary categorical distribution | —                              | `outputs/figures/categorical_dirichlet_bernoulli.png` |

For example:

```bash
python -m chapters.ch02 2  # Bernoulli frequentist and Bayesian estimates
python -m chapters.ch02 4  # Categorical, Dirichlet, and Bernoulli comparison
```

Individual scripts can also be run directly:

```bash
python -m chapters.ch02.figure_2_2
python -m chapters.ch02.bernoulli_frequentisist_bayesian
python -m chapters.ch02.figure_2_3
python -m chapters.ch02.categorical_dirichlet_bernoulli
```

## Shared implementation

The experiments reuse the distribution classes under
`src/prml/distributions/`:

- `Beta` evaluates the beta density and acts as the conjugate prior for a
  Bernoulli probability.
- `Bernoulli` supports probability evaluation, sampling, maximum-likelihood
  estimation, and Bayesian updating with a beta prior.
- `Dirichlet` evaluates and samples distributions over probability vectors and
  acts as the conjugate prior for categorical probabilities.
- `Categorical` supports one-hot observations, probability evaluation,
  sampling, maximum-likelihood estimation, and Bayesian updating with a
  Dirichlet prior.
