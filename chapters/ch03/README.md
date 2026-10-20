# Chapter 3: Linear Models for Regression

Reproductions and demonstrations of the linear regression models from Chapter 3
of Christopher M. Bishop's _Pattern Recognition and Machine Learning_ (PRML).

## Running the experiments

Install the project from the repository root, then invoke the Chapter 3 module
with an experiment number, take 1 for example:

```bash
python -m chapters.ch03 1
```

Run every implemented Chapter 3 experiment with:

```bash
python -m chapters.ch03 all
```

Each experiment reads its parameters from `configs/ch03/` and writes its plot
to `outputs/figures/ch03/`.

## Implemented figures

| Experiment | Figure or demonstration | Concept                                                                | Configuration                                      | Output                                                    |
| ---------: | :---------------------: | ---------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
|        `1` |       Figure 3.1        | Polynomial, Gaussian, and sigmoidal basis functions                    | `configs/ch03/figure_3_1.yaml`                     | `outputs/figures/ch03/figure_3_1.png`                     |
|        `2` |    Figure 3.2 plus      | Least-squares regression with three families of basis functions        | `configs/ch03/figure_3_2.yaml`                     | `outputs/figures/ch03/figure_3_2.png`                     |
|        `3` | Sequential learning demo | Online linear regression with sequential parameter updates             | `configs/ch03/figure_sequential_learning.yaml`     | `outputs/figures/ch03/figure_sequential_learning.png`     |
|        `4` | Ridge regression demo   | Regularized regression with polynomial, Gaussian, and sigmoid bases    | `configs/ch03/ridge_regression.yaml`               | `outputs/figures/ch03/ridge_regression.png`               |
|        `5` |       Figure 3.5        | Effect of regularization on fitted curves and their ensemble average   | `configs/ch03/figure_3_5.yaml`                     | `outputs/figures/ch03/figure_3_5.png`                     |
|        `6` |       Figure 3.7        | Sequential Bayesian linear regression in weight and data space         | `configs/ch03/figure_3_7.yaml`                     | `outputs/figures/ch03/figure_3_7.png`                     |
|        `7` |       Figure 3.8        | Predictive mean and uncertainty as observations accumulate             | `configs/ch03/figure_3_8.yaml`                     | `outputs/figures/ch03/figure_3_8.png`                     |
|        `8` |       Figure 3.9        | Functions sampled from successive Bayesian posterior distributions     | `configs/ch03/figure_3_9.yaml`                     | `outputs/figures/ch03/figure_3_9.png`                     |
|        `9` |       Figure 3.10       | Equivalent kernel for Bayesian linear regression                       | `configs/ch03/figure_3_10.yaml`                    | `outputs/figures/ch03/figure_3_10.png`                    |

For example:

```bash
python -m chapters.ch03 6  # Figure 3.7
python -m chapters.ch03 9  # Figure 3.10
```

## Shared implementation

The experiment scripts reuse the numerical components under
`src/prml/linear_models/`:

- `polynomial_basis`, `gaussian_basis`, and `sigmoidal_basis` construct design
  matrices from the corresponding families of basis functions.
- `LinearRegression` fits an unregularized least-squares model.
- `RidgeRegression` minimizes the sum-of-squares objective with an $L_2$
  penalty controlled by the regularization coefficient.
- `SequentialLinearRegression` updates a linear model one observation at a
  time using stochastic gradient descent.
- `BayesianLinearRegression` performs sequential posterior updates, evaluates
  predictive distributions, and samples weights from the posterior.

The configurations can be edited to explore different random seeds, noise
levels, sample sizes, basis functions, and regularization strengths without
changing the experiment code.
