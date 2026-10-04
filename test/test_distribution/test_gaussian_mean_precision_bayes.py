import numpy as np
import pytest

from prml.distributions.gaussian import GaussianMeanPrecisionBayes
from prml.distributions.normal_gamma import NormalGamma


def test_requires_normal_gamma_prior():
    with pytest.raises(TypeError, match="prior"):
        GaussianMeanPrecisionBayes(prior="not a NormalGamma")


def test_fit_returns_normal_gamma():
    prior = NormalGamma.form_constraints(
        c=2.0,
        beta=4.0,
        d=3.0,
    )
    model = GaussianMeanPrecisionBayes(prior=prior)

    posterior = model.fit([1.0, 2.0, 4.0])

    assert isinstance(posterior, NormalGamma)


def test_fit_updates_constraint_parameters():
    prior = NormalGamma.form_constraints(
        c=2.0,
        beta=4.0,
        d=3.0,
    )
    model = GaussianMeanPrecisionBayes(prior=prior)

    observations = np.array([1.0, 2.0, 4.0])
    posterior = model.fit(observations)

    expected_c = 2.0 + observations.sum()
    expected_beta = 4.0 + observations.size
    expected_d = 3.0 + 0.5 * np.sum(observations**2)

    assert posterior.c == pytest.approx(expected_c)
    assert posterior.beta == pytest.approx(expected_beta)
    assert posterior.d == pytest.approx(expected_d)
