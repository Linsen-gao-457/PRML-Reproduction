import numpy as np
import pytest
from scipy.stats import t

from prml.distributions.student_t import Student_T


def test_initialization():
    distribution = Student_T(
        mean=1.0,
        precision=2.0,
        degree_of_freedom=3.0,
    )

    assert distribution.mean == pytest.approx(1.0)
    assert distribution.precision == pytest.approx(2.0)
    assert distribution.degree_of_freedom == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("mean", np.nan),
        ("mean", np.inf),
        ("precision", 0.0),
        ("precision", -1.0),
        ("precision", np.nan),
        ("degree_of_freedom", 0.0),
        ("degree_of_freedom", -1.0),
        ("degree_of_freedom", np.inf),
    ],
)
def test_rejects_invalid_parameters(parameter, value):
    arguments = {
        "mean": 0.0,
        "precision": 1.0,
        "degree_of_freedom": 3.0,
    }
    arguments[parameter] = value

    with pytest.raises((TypeError, ValueError)):
        Student_T(**arguments)


def test_cauchy_density_at_location():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=1.0,
    )

    density = distribution.pdf(0.0)

    assert density == pytest.approx(1.0 / np.pi)


def test_pdf_returns_scalar_for_scalar_input():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=3.0,
    )

    assert np.isscalar(distribution.pdf(0.0))


def test_pdf_returns_array_for_array_input():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=3.0,
    )
    x = np.array([-1.0, 0.0, 1.0])

    density = distribution.pdf(x)

    assert density.shape == (3,)
    assert np.all(density > 0)


def test_pdf_is_symmetric_about_mean():
    distribution = Student_T(
        mean=2.0,
        precision=1.5,
        degree_of_freedom=4.0,
    )

    assert distribution.pdf(1.0) == pytest.approx(distribution.pdf(3.0))


def test_pdf_matches_scipy():
    distribution = Student_T(
        mean=1.5,
        precision=2.0,
        degree_of_freedom=4.0,
    )
    x = np.linspace(-3.0, 6.0, 100)

    actual = distribution.pdf(x)
    expected = t.pdf(
        x,
        df=4.0,
        loc=1.5,
        scale=1.0 / np.sqrt(2.0),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_pdf_integrates_to_one():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=5.0,
    )
    x = np.linspace(-100.0, 100.0, 100_000)

    integral = np.trapezoid(distribution.pdf(x), x)

    assert integral == pytest.approx(1.0, abs=1e-5)


def test_draw_returns_reproducible_samples_with_correct_shape():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=3.0,
    )

    first = distribution.draw(10, rng=np.random.default_rng(123))
    second = distribution.draw(10, rng=np.random.default_rng(123))

    assert first.shape == (10,)
    np.testing.assert_allclose(first, second)


def test_expectation_known_values():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=3.0,
    )

    expected_eta, expected_log_eta = distribution._expectation(np.array([0.0, 1.0]))

    np.testing.assert_allclose(expected_eta, [4.0 / 3.0, 1.0])
    assert expected_log_eta.shape == (2,)
    assert np.all(np.isfinite(expected_log_eta))


def test_outlier_receives_smaller_weight():
    distribution = Student_T(
        mean=0.0,
        precision=1.0,
        degree_of_freedom=3.0,
    )

    expected_eta, _ = distribution._expectation(np.array([0.1, 10.0]))

    assert expected_eta[1] < expected_eta[0]


@pytest.mark.parametrize(
    "observations",
    [
        [],
        [1.0],
        [[1.0, 2.0]],
        [1.0, np.nan],
        [1.0, np.inf],
        ["a", "b"],
    ],
)
def test_fit_rejects_invalid_observations(observations):
    distribution = Student_T()

    with pytest.raises((TypeError, ValueError)):
        distribution.fit(observations)
