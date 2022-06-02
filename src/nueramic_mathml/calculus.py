# Simple common math functions for show examples
# Gradient, Hessian, Jacobian
# Radial basis functions
# Logistic function

import torch
from typing import Callable


def gradient(function: Callable[[torch.Tensor], float], x0: torch.Tensor, delta_x: float = 1e-8) -> torch.Tensor:
    """
    Returns the gradient of the function at a specific point x0
    A two-point finite difference formula that approximates the derivative

    .. math::

        \\displaystyle \\frac{\\partial f}{\\partial x} \\approx {\\frac {f(x+h)-f(x-h)}{2h}}

    Gradient

    .. math::

         \\displaystyle \\nabla f = \\left[\\frac{\\partial f}{\\partial x_1} \\enspace
         \\frac{\\partial f}{\\partial x_2}
         \\enspace \\dots \\enspace \\frac{\\partial f}{\\partial x_n}\\right]^\\top

    :param function: function which depends on n variables from x
    :param x0: n x 1 - dimensional array :math:`\\in R^{n}`
    :param delta_x: precision of two-point formula above (delta_x = h)
    :return: vector of partial derivatives
    """
    x0 = x0.flatten()
    grad = torch.zeros_like(x0)
    assert not isinstance(x0, torch.Tensor), 'x0 must be torch.Tensor'

    for i in range(len(x0)):
        delta = torch.zeros_like(x0, dtype=torch.float64)
        delta[i] += delta_x
        grad_i = (function(x0 + delta) - function(x0 - delta)) / (2 * delta_x)
        grad[i] = grad_i

    return grad
