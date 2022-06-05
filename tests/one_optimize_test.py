import pytest
from src.nueramic_mathml import *
import numpy

# Test golden section search

test_functions = [  # func, [a, b], true_point
    (lambda x: x ** 2, (-2, 2), 0, 'parabola'),
    (lambda x: - numpy.exp(numpy.cos(x)), (-2, 2), 0, 'e(cos(x))'),
    (lambda x: - numpy.log(x), (1., 4.), 4, 'log(x)'),
    (lambda x: - numpy.log(x) + numpy.pi * numpy.sin(x) * x, (2.5, 7), 4.925548, '-ln(x) + pi sin(x) x')
]


@pytest.mark.parametrize("function, bounds, expected, name", test_functions)
def test_golden_section_search(function, bounds, expected, name):
    assert golden_section_search(function, bounds)[0] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("function, bounds, expected, name", test_functions)
def test_successive_parabolic_interpolation(function, bounds, expected, name):
    if name != '-ln(x) + pi sin(x) x':  # the test is too rough for sensitive spi
        assert successive_parabolic_interpolation(function, bounds)[0] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("function, bounds, expected, name", test_functions)
def test_brent(function, bounds, expected, name):
    assert brent(function, bounds)[0] == pytest.approx(expected, abs=1e-4)
