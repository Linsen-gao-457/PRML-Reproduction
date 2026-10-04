import numpy as np
import pytest
from scipy.stats import norm

from prml.distributions.gaussian import Gaussian, GaussianMeanBayes


def test_initialization():
    gaussian = Gaussian(
        mean=1.5,
        standard_deviation=2.0,
    )

    assert gaussian.mean == 1.5
    assert gaussian.standard_deviation == 2.0


def test_numpy_scalar_parameters():
    gaussian = Gaussian(
        mean=np.float64(1.5),
        standard_deviation=np.float64(2.0),
    )

    assert gaussian.mean == 1.5
    assert gaussian.standard_deviation == 2.0


def test_pdf_at_mean():
    gaussian = Gaussian(
        mean=1.5,
        standard_deviation=2.0,
    )

    expected = 1.0 / (np.sqrt(2.0 * np.pi) * gaussian.standard_deviation)

    assert gaussian.pdf(gaussian.mean) == pytest.approx(expected)


def test_pdf_is_symmetric():
    gaussian = Gaussian(
        mean=1.5,
        standard_deviation=2.0,
    )

    distance = 0.75

    left = gaussian.pdf(gaussian.mean - distance)
    right = gaussian.pdf(gaussian.mean + distance)

    assert left == pytest.approx(right)


def test_pdf_array_shape():
    gaussian = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )
    x = np.linspace(-3.0, 3.0, 20)

    density = gaussian.pdf(x)

    assert density.shape == x.shape
    assert np.all(density >= 0.0)


def test_pdf_matches_scipy():
    mean = 1.5
    standard_deviation = 2.0

    gaussian = Gaussian(
        mean=mean,
        standard_deviation=standard_deviation,
    )

    x = np.array([-3.0, -1.0, 0.0, 1.5, 2.0, 4.0])

    expected = norm.pdf(
        x,
        loc=mean,
        scale=standard_deviation,
    )

    np.testing.assert_allclose(
        gaussian.pdf(x),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pdf_integrates_to_one():
    gaussian = Gaussian(
        mean=1.5,
        standard_deviation=2.0,
    )

    x = np.linspace(
        gaussian.mean - 8 * gaussian.standard_deviation,
        gaussian.mean + 8 * gaussian.standard_deviation,
        20_001,
    )
    density = gaussian.pdf(x)

    area = np.trapz(density, x)

    assert area == pytest.approx(1.0, abs=1e-6)


def test_draw_shape():
    gaussian = Gaussian(
        mean=1.5,
        standard_deviation=2.0,
    )
    rng = np.random.default_rng(1234)

    samples = gaussian.draw(100, rng=rng)

    assert samples.shape == (100,)


def test_draw_is_reproducible():
    gaussian = Gaussian(
        mean=1.5,
        standard_deviation=2.0,
    )

    first = gaussian.draw(
        20,
        rng=np.random.default_rng(1234),
    )
    second = gaussian.draw(
        20,
        rng=np.random.default_rng(1234),
    )

    np.testing.assert_array_equal(first, second)


def test_empirical_mean_and_standard_deviation():
    gaussian = Gaussian(
        mean=3.0,
        standard_deviation=4.0,
    )
    rng = np.random.default_rng(1234)

    samples = gaussian.draw(200_000, rng=rng)

    assert samples.mean() == pytest.approx(
        gaussian.mean,
        abs=0.04,
    )
    assert samples.std() == pytest.approx(
        gaussian.standard_deviation,
        abs=0.04,
    )


@pytest.mark.parametrize(
    "mean",
    [
        [0.0],
        np.array([0.0]),
        "0.0",
    ],
)
def test_invalid_mean_type(mean):
    with pytest.raises(TypeError):
        Gaussian(
            mean=mean,
            standard_deviation=1.0,
        )


@pytest.mark.parametrize(
    "mean",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_mean(mean):
    with pytest.raises(ValueError):
        Gaussian(
            mean=mean,
            standard_deviation=1.0,
        )


@pytest.mark.parametrize(
    "standard_deviation",
    [
        [1.0],
        np.array([1.0]),
        "1.0",
    ],
)
def test_invalid_standard_deviation_type(standard_deviation):
    with pytest.raises(TypeError):
        Gaussian(
            mean=0.0,
            standard_deviation=standard_deviation,
        )


@pytest.mark.parametrize(
    "standard_deviation",
    [
        0.0,
        -1.0,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_standard_deviation_value(standard_deviation):
    with pytest.raises(ValueError):
        Gaussian(
            mean=0.0,
            standard_deviation=standard_deviation,
        )


def test_fit_returns_self():
    gaussian = Gaussian()

    result = gaussian.fit([1.0, 2.0, 3.0])

    assert result is gaussian


def test_fit_known_values():
    gaussian = Gaussian()

    gaussian.fit([1.0, 2.0, 3.0])

    expected_mean = 2.0
    expected_standard_deviation = np.sqrt(1)

    assert gaussian.mean == pytest.approx(expected_mean)
    assert gaussian.standard_deviation == pytest.approx(expected_standard_deviation)


def test_gaussian_mean_bayes_fit_known_values():
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )

    model = GaussianMeanBayes(
        prior=prior,
        precision=1.0,
    )

    posterior = model.fit([2.0, 4.0])

    assert posterior.mean == pytest.approx(2.0)
    assert posterior.standard_deviation == pytest.approx(np.sqrt(1.0 / 3.0))


def test_gaussian_mean_bayes_fit_returns_gaussian():
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )

    model = GaussianMeanBayes(
        prior=prior,
        precision=1.0,
    )

    posterior = model.fit([2.0, 4.0])

    assert isinstance(posterior, Gaussian)


def test_gaussian_mean_bayes_does_not_modify_prior():
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )

    model = GaussianMeanBayes(
        prior=prior,
        precision=1.0,
    )

    model.fit([2.0, 4.0])

    assert prior.mean == 0.0
    assert prior.standard_deviation == 1.0


def test_gaussian_mean_bayes_accepts_single_observation():
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )

    model = GaussianMeanBayes(
        prior=prior,
        precision=1.0,
    )

    posterior = model.fit([2.0])

    assert posterior.mean == pytest.approx(1.0)
    assert posterior.standard_deviation == pytest.approx(np.sqrt(1.0 / 2.0))


@pytest.mark.parametrize(
    "observations",
    [
        [],
        [[1.0, 2.0]],
        [[1.0], [2.0]],
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0, -np.inf],
    ],
)
def test_gaussian_mean_bayes_rejects_invalid_observations(observations):
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )
    model = GaussianMeanBayes(
        prior=prior,
        precision=1.0,
    )

    with pytest.raises((TypeError, ValueError)):
        model.fit(observations)


@pytest.mark.parametrize(
    "observations",
    [
        ["a", "b"],
        [1.0, "a"],
    ],
)
def test_gaussian_mean_bayes_rejects_non_numeric_observations(observations):
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )
    model = GaussianMeanBayes(
        prior=prior,
        precision=1.0,
    )

    with pytest.raises(TypeError):
        model.fit(observations)


def test_stronger_observation_precision_moves_posterior_toward_data():
    prior = Gaussian(
        mean=0.0,
        standard_deviation=1.0,
    )

    weak_data = GaussianMeanBayes(
        prior=prior,
        precision=0.1,
    )

    strong_data = GaussianMeanBayes(
        prior=prior,
        precision=10.0,
    )

    weak_posterior = weak_data.fit([10.0])
    strong_posterior = strong_data.fit([10.0])

    assert strong_posterior.mean > weak_posterior.mean
