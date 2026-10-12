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
under `outputs/figures/ch02/`.

## Implemented experiments

| Experiment | Figure or demonstration | Concept                                                                                           | Configuration                   | Output                                                     |
| ---------: | :---------------------: | ------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------- |
|        `1` |       Figure 2.2        | Beta densities for several values of $a$ and $b$                                                  | `configs/ch02/figure_2_2.yaml`  | `outputs/figures/ch02/figure_2_2.png`                      |
|        `2` |     Bernoulli demo      | Frequentist estimation and beta-Bernoulli Bayesian updating                                       | —                               | Terminal output                                            |
|        `3` |       Figure 2.3        | One step of sequential Bayesian inference for Bernoulli data                                      | `configs/ch02/figure_2_3.yaml`  | `outputs/figures/ch02/figure_2_3.png`                      |
|        `4` |    Categorical demo     | Categorical MLE, Dirichlet posterior updating, and Bernoulli as a binary categorical distribution | —                               | `outputs/figures/ch02/categorical_dirichlet_bernoulli.png` |
|        `5` |       Figure 2.6        | Sampling distribution of the mean for uniform observations                                        | `configs/ch02/figure_2_6.yaml`  | `outputs/figures/ch02/figure_2_6.png`                      |
|        `6` |       Figure 2.12       | Bayesian inference for a Gaussian mean with known precision                                       | `configs/ch02/figure_2_12.yaml` | `outputs/figures/ch02/figure_2_12.png`                     |
|        `7` |    Figure 2.13 plus     | Bayesian inference for Gaussian precision with a known mean                                       | `configs/ch02/figure_2_13.yaml` | `outputs/figures/ch02/figure_2_13_plus.png`                |
|        `8` |    Figure 2.14 plus     | Normal–Gamma prior and posterior contours for unknown Gaussian mean and precision                 | `configs/ch02/figure_2_14.yaml` | `outputs/figures/ch02/figure_2_14_plus.png`                |
|        `9` |       Figure 2.15       | Student's t densities for different degrees of freedom and the Gaussian limit                     | `configs/ch02/figure_2_15.yaml` | `outputs/figures/ch02/figure_2_15.png`                     |
|       `10` |       Figure 2.16       | Robustness of Student's t fitting compared with Gaussian fitting in the presence of outliers      | `configs/ch02/figure_2_16.yaml` | `outputs/figures/ch02/figure_2_16.png`                     |
|       `11` |       Figure 2.19       | Cartesian and circular-coordinate views of von Mises densities                                    | `configs/ch02/figure_2_19.yaml` | `outputs/figures/ch02/figure_2_19.png`                     |
|       `12` |       Figure 2.24       | Histogram density estimation for several bin widths                                               | `configs/ch02/figure_2_24.yaml` | `outputs/figures/ch02/figure_2_24.png`                     |
|       `13` |       Figure 2.25       | Gaussian kernel density estimation for several bandwidths                                         | `configs/ch02/figure_2_25.yaml` | `outputs/figures/ch02/figure_2_25.png`                     |
|       `14` |       Figure 2.26       | K-nearest-neighbour density estimation for several values of $K$                                  | `configs/ch02/figure_2_26.yaml` | `outputs/figures/ch02/figure_2_26.png`                     |

For example:

```bash
python -m chapters.ch02 2  # Bernoulli frequentist and Bayesian estimates

```

Individual scripts can also be run directly:

```bash
python -m chapters.ch02.figure_2_2
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
- `Gaussian` supports density evaluation, sampling, and maximum-likelihood
  estimation, with companion classes for Bayesian inference over its mean,
  precision, or both.
- `Gamma` and `NormalGamma` provide conjugate priors for Gaussian precision
  and joint mean–precision inference.
- `Student_T` evaluates, samples, and fits Student's t distributions, allowing
  the robustness comparison in Figure 2.16.
- `VonMises` contains the parameter representation for periodic angular
  variables; Figure 2.19 evaluates the density directly for its two plot
  styles.

Figures 2.24–2.26 reuse the same seeded Gaussian-mixture sample to compare
histogram, kernel, and K-nearest-neighbour approaches to density estimation.
