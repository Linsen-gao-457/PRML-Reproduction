# Chapter 1: Introduction

Reproductions of the polynomial curve-fitting figures from Chapter 1 of
Christopher M. Bishop's _Pattern Recognition and Machine Learning_ (PRML).

## Running the experiments

Install the project from the repository root, then invoke the Chapter 1 module
with an experiment number, take 1 for example:

```bash
python -m chapters.ch01 1
```

Run every implemented Chapter 1 experiment with:

```bash
python -m chapters.ch01 all
```

Each experiment reads its parameters from `configs/ch01/` and writes its plot
to `outputs/figures/`.

## Implemented figures

| Experiment | Figure | Concept                                                       | Configuration                  | Output                           |
| ---------: | :----: | ------------------------------------------------------------- | ------------------------------ | -------------------------------- |
|        `1` |  1.2   | Samples from $\sin(2\pi x)$ with additive Gaussian noise      | `configs/ch01/figure_1_2.yaml` | `outputs/figures/figure_1_2.png` |
|        `2` |  1.4   | Polynomial least-squares fits for $M=0,1,3,9$                 | `configs/ch01/figure_1_4.yaml` | `outputs/figures/figure_1_4.png` |
|        `3` |  1.5   | Training and test RMS error versus polynomial degree          | `configs/ch01/figure_1_5.yaml` | `outputs/figures/figure_1_5.png` |
|        `4` |  1.6   | Effect of increasing the training-set size for an $M=9$ model | `configs/ch01/figure_1_6.yaml` | `outputs/figures/figure_1_6.png` |
|        `5` |  1.7   | Degree-9 ridge fits for two regularization strengths          | `configs/ch01/figure_1_7.yaml` | `outputs/figures/figure_1_7.png` |
|        `6` |  1.8   | Training and test RMS error versus $\ln\lambda$               | `configs/ch01/figure_1_8.yaml` | `outputs/figures/figure_1_8.png` |

For example:

```bash
python -m chapters.ch01 3  # Figure 1.5
python -m chapters.ch01 6  # Figure 1.8
```

## Shared implementation

The figure scripts reuse the numerical components under `src/prml/`:

- `linear_models.polynomial_basis` constructs the design matrix
  $[1,x,x^2,\ldots,x^M]$.
- `linear_models.LinearRegression` fits an unregularized least-squares model.
- `linear_models.RidgeRegression` minimizes the sum-of-squares objective with
  an $L_2$ penalty controlled by $\lambda$.
- `utils.random.create_rng` makes generated datasets reproducible from the YAML
  seed.

The configurations can be edited to explore different seeds, noise levels,
sample sizes, polynomial degrees, and regularization strengths without changing
the experiment code.
