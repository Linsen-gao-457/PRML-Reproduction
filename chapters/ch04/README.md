# Chapter 4: Linear Models for Classification

Reproductions and demonstrations of the classification models from Chapter 4
of Christopher M. Bishop's _Pattern Recognition and Machine Learning_ (PRML).

## Running the experiments

Install the project from the repository root, then invoke the Chapter 4 module
with an experiment number. For example:

```bash
python -m chapters.ch04 1
```

Run every implemented Chapter 4 experiment with:

```bash
python -m chapters.ch04 all
```

Each experiment reads its parameters from `configs/ch04/` and writes its plot
to `outputs/figures/ch04/` by default. Output paths can be changed in the YAML
configurations.

## Implemented figures

| Experiment |       Figure or demonstration       | Concept                                                                                           | Configuration                                      | Output                                                    |
| ---------: | :---------------------------------: | ------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
|        `1` |             Figure 4.4              | Effect of outliers on least-squares classification and logistic regression                        | `configs/ch04/figure_4_4.yaml`                     | `outputs/figures/ch04/figure_4_4.png`                     |
|        `2` |             Figure 4.5              | Three-class decision regions for least-squares classification and multinomial logistic regression | `configs/ch04/figure_4_5.yaml`                     | `outputs/figures/ch04/figure_4_5.png`                     |
|        `3` |             Figure 4.6              | Projection onto the line joining class means compared with Fisher's linear discriminant           | `configs/ch04/figure_4_6.yaml`                     | `outputs/figures/ch04/figure_4_6.png`                     |
|        `4` |             Figure 4.7              | Sequential perceptron updates and the resulting decision boundaries                               | `configs/ch04/figure_4_7.yaml`                     | `outputs/figures/ch04/figure_4_7.png`                     |
|        `5` | Gaussian generative classifier demo | Multiclass Gaussian classification, posterior confidence, and sampling from the fitted model      | `configs/ch04/gaussian_generative_classifier.yaml` | `outputs/figures/ch04/gaussian_generative_classifier.png` |
|        `6` |       Probit activation demo        | Gaussian threshold probability and comparison with the logistic sigmoid                           | `configs/ch04/probit_activation.yaml`              | `outputs/figures/ch04/probit_activation.png`              |

For example:

```bash
python -m chapters.ch04 3  # Figure 4.6
python -m chapters.ch04 5  # Gaussian generative classifier demo
```

Individual scripts can also be run directly as modules:

```bash
python -m chapters.ch04.gaussian_generative_classifier
```

## Shared implementation

The experiment scripts reuse the numerical components under
`src/prml/classification/`:

- `LeastSquaresClassifier` fits linear class scores using least squares.
- `MulticlassLogisticRegression` learns multiclass conditional probabilities
  using the softmax function and also supports two classes.
- `FisherLinearDiscriminant` finds a projection that separates two classes
  relative to their within-class variation.
- `PerceptronClassifier` learns a linear decision boundary through sequential
  updates on misclassified observations.
- `GaussianGenerativeClassifier` estimates class priors, class means, and a
  shared covariance matrix. It computes posterior probabilities using Bayes'
  rule, predicts class labels, and generates labeled samples from the fitted
  distributions.

The configurations can be edited to explore different random seeds, sample
sizes, class means, covariance matrices, model settings, and plot styles
without changing the experiment code.

For the Gaussian generative classifier demo, the YAML sections control:

- `data`: class means, shared covariance, class priors, training and test
  sample counts, and their random seeds.
- `model`: covariance diagonal regularization through `reg`.
- `sampling`: the number of generated observations and their random seed.
- `plot`: figure size, grid resolution, colors, marker sizes, transparency,
  confidence levels, and output resolution.
- `output`: the saved figure path, relative to the repository root.

The demo plots two input features. To change the number of classes, update
the rows in `data.means`, provide matching `data.priors` that sum to one, and
configure at least one plot color per class. Use enough training observations
to represent every configured class.

For the probit activation demo, use `plot.score_range` and
`plot.example_score` to choose the horizontal range and the score used to
shade the Gaussian threshold probability. The demo plots the curves without
fitting a regression model.
