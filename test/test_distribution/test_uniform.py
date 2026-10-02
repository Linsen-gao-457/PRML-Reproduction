import numpy as np
import pytest

from prml.distributions.uniform import Uniform


def test_initialization():
    uniform = Uniform(low=0.0, high=2.0)

    assert uniform.low == 0.0
    assert uniform.high == 2.0


def test_pdf_scalar_inside():
    uniform = Uniform(low=0.0, high=2.0)

    assert np.isclose(uniform._pdf(1.0), 0.5)


def test_pdf_scalar_outside():
    uniform = Uniform(low=0.0, high=2.0)

    assert uniform._pdf(-1.0) == 0.0
    assert uniform._pdf(3.0) == 0.0


def test_pdf_array():
    uniform = Uniform(low=0.0, high=2.0)

    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    expected = np.array([0.0, 0.5, 0.5, 0.5, 0.0])

    assert np.allclose(uniform._pdf(x), expected)


def test_invalid_interval():
    with pytest.raises(ValueError):
        Uniform(low=1.0, high=1.0)

    with pytest.raises(ValueError):
        Uniform(low=2.0, high=1.0)


def test_invalid_low():
    with pytest.raises(TypeError):
        Uniform(low=[0], high=1)

    with pytest.raises(ValueError):
        Uniform(low=np.inf, high=1)


def test_invalid_high():
    with pytest.raises(TypeError):
        Uniform(low=0, high=[1])

    with pytest.raises(ValueError):
        Uniform(low=0, high=np.inf)


def test_draw():
    uniform = Uniform(low=0.0, high=2.0)
    rng = np.random.default_rng(42)

    samples = uniform._draw(1000, rng)

    assert samples.shape == (1000,)
    assert np.all(samples >= 0.0)
    assert np.all(samples < 2.0)
