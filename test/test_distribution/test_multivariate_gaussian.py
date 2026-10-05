import numpy as np
import pytest

from prml.distributions.multivariate_gaussian import MultivariateGaussian


def test_initialization_calculates_inverse_covariance():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])

    distribution = MultivariateGaussian(mean=[1.0, 2.0], cov=covariance)

    np.testing.assert_allclose(distribution.mean, [1.0, 2.0])
    np.testing.assert_allclose(distribution.cov, covariance)
    np.testing.assert_allclose(
        distribution.inv_cov,
        np.linalg.inv(covariance),
    )
    np.testing.assert_allclose(
        distribution.cov @ distribution.inv_cov,
        np.eye(2),
    )


def test_pdf_at_mean_for_two_dimensional_standard_gaussian():
    distribution = MultivariateGaussian(mean=[0.0, 0.0], cov=np.eye(2))

    density = distribution.pdf([0.0, 0.0])

    assert density == pytest.approx(1.0 / (2.0 * np.pi))


def test_pdf_accepts_multiple_points():
    distribution = MultivariateGaussian(mean=[0.0, 0.0], cov=np.eye(2))
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    density = distribution.pdf(points)

    expected = np.array(
        [
            1.0 / (2.0 * np.pi),
            np.exp(-0.5) / (2.0 * np.pi),
            np.exp(-0.5) / (2.0 * np.pi),
        ]
    )
    assert density.shape == (3,)
    np.testing.assert_allclose(density, expected)


def test_pdf_is_highest_at_mean():
    distribution = MultivariateGaussian(mean=[1.0, 2.0], cov=np.eye(2))

    assert distribution.pdf([1.0, 2.0]) > distribution.pdf([2.0, 3.0])


def test_draw_returns_reproducible_samples_with_correct_shape():
    distribution = MultivariateGaussian(mean=[0.0, 0.0], cov=np.eye(2))

    first = distribution.draw(10, rng=np.random.default_rng(123))
    second = distribution.draw(10, rng=np.random.default_rng(123))

    assert first.shape == (10, 2)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    ("mean", "cov", "message"),
    [
        ([[0.0, 1.0]], np.eye(2), "one-dimensional"),
        ([0.0, np.nan], np.eye(2), "finite"),
        ([0.0, 0.0], np.ones((2, 3)), "square"),
        ([0.0, 0.0], [[1.0, 1.0], [0.0, 1.0]], "symmetric"),
        ([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], "positive definite"),
        ([0.0, 0.0], np.eye(3), "dimension"),
    ],
)
def test_rejects_invalid_parameters(mean, cov, message):
    with pytest.raises(ValueError, match=message):
        MultivariateGaussian(mean=mean, cov=cov)


def test_pdf_rejects_wrong_point_dimension():
    distribution = MultivariateGaussian(mean=[0.0, 0.0], cov=np.eye(2))

    with pytest.raises(ValueError, match="shape"):
        distribution.pdf([0.0, 1.0, 2.0])


def test_pdf_and_draw_require_known_parameters():
    distribution = MultivariateGaussian()

    with pytest.raises(RuntimeError, match="mean"):
        distribution.pdf([0.0, 0.0])
    with pytest.raises(RuntimeError, match="mean"):
        distribution.draw(1)
