# Simple common math functions for show examples
# Gradient, Hessian, Jacobian

from __future__ import annotations

from typing import Callable, Sequence

import torch


def gradient(function: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, delta_x: float = 1e-4) -> torch.Tensor:
    """
    Returns the gradient of the function at a specific point :math:`x_0`.
    A two-point finite difference formula that approximates the derivative

    .. math::

        \\displaystyle \\frac{\\partial f}{\\partial x} \\approx {\\frac {f(x+h)-f(x-h)}{2h}}

    Gradient

    .. math::

         \\displaystyle \\nabla f = \\left[\\frac{\\partial f}{\\partial x_1} \\enspace
         \\frac{\\partial f}{\\partial x_2}
         \\enspace \\dots \\enspace \\frac{\\partial f}{\\partial x_n}\\right]^\\top

    Example
        >>> gradient(lambda x: (x ** 2).sum(), torch.tensor([1., 2.])).round()  # f(x, y)  = x ** 2 + y ** 2
            tensor([2., 4.], dtype=torch.float64)

    :param function: function which depends on n variables from x
    :param x0: n x 1 - dimensional array :math:`\\in \\mathbb{R}^{n}`. dtype is torch.double (float64)
    :param delta_x: precision of two-point formula above (delta_x = h)
    :return: vector of partial derivatives

    .. note::
        If we make delta_x :math:`\\leq` 1e-4 gradient will return values with large error rate

    """
    assert isinstance(x0, torch.Tensor), 'x0 must be torch.Tensor'

    x0: torch.Tensor = x0.flatten().double()
    delta_x = torch.tensor(max(delta_x, 1e-4), dtype=x0.dtype)
    grad = torch.zeros_like(x0, dtype=x0.dtype)

    for i in range(len(x0)):
        delta = torch.zeros_like(x0, dtype=x0.dtype)
        delta[i] += delta_x
        grad_i = (function(x0 + delta) - function(x0 - delta)) / (2 * delta_x)
        grad[i] = grad_i

    return grad


def jacobian(f_vector: Sequence[Callable[[torch.Tensor], torch.Tensor]],
             x0: torch.Tensor,
             delta_x: float = 1e-4) -> torch.Tensor:
    """
    Returns the Jacobian matrix of a sequence of m functions from f_vector by n variables from x.

    .. math::

        {\\displaystyle J ={\\begin{bmatrix}{\\dfrac {\\partial f_{1}}{\\partial x_{1}}}&\\cdots
          &{\\dfrac {\\partial f_{1}}{\\partial x_{n}}}\\\\\\vdots &\\ddots &\\vdots \\\\{\\dfrac {\\partial f_{m}}{\\partial x_{1}}}
          &\\cdots &{\\dfrac {\\partial f_{m}}{\\partial x_{n}}}\\end{bmatrix}}}_{m \\times n}


    >>> func_3 = [lambda x: x[0] ** 2 + x[1], lambda x: 2 * x[0] + 5 * x[1], lambda x: x[0] * x[1]]
    >>> print(jacobian(func_3, torch.tensor([-1, 2])).round())
    tensor([[-2.,  1.],
            [ 2.,  5.],
            [ 2., -1.]], dtype=torch.float64)

    :param f_vector: a flat sequence, list or tuple or other containing m functions
    :param x0: an n-dimensional array. The specific point at which we will calculate the Jacobian
    :param delta_x: precision of gradient
    :return: the Jacobian matrix according to the above formula. Matrix n x m
    """
    assert isinstance(f_vector, Sequence), 'f_vector must be sequence'
    jac = torch.zeros(len(f_vector), x0.flatten().shape[0], dtype=torch.float64)
    for j in range(len(f_vector)):
        jac[j, :] = gradient(f_vector[j], x0, delta_x)

    return jac


def hessian(function: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, delta_x: float = 1e-4) -> torch.Tensor:
    """
    Returns a hessian of function at point :math:`x_0`


    .. math::
        \\ H(f) = \\begin{bmatrix} \\displaystyle
        \\frac{\\partial^2 f}{\\partial x_1^2} & \\displaystyle \\frac{\\partial^2 f}{\\partial x_1\\,\\partial x_2} &
        \\cdots & \\displaystyle \\frac{\\partial^2 f}{\\partial x_1\\,\\partial x_n} \\\\  \\
        \\displaystyle \\frac{\\partial^2 f}{\\partial x_2\\,\\partial x_1} & \\displaystyle \\frac{\\partial^2 f}
        {\\partial x_2^2} & \\cdots & \\displaystyle \\frac{\\partial^2 f}{\\partial x_2\\,\\partial x_n} \\\\  \\
        \\vdots & \\vdots & \\ddots & \\vdots \\\\  \\
        \\displaystyle \\frac{\\partial^2 f}{\\partial x_n\\,\\partial x_1} & \\displaystyle \\frac{\\partial^2 f}
        {\\partial x_n\\,\\partial x_2} & \\cdots & \\displaystyle \\frac{\\partial^2 f}{\\partial x_n^2}
        \\end{bmatrix}\\\\



    >>> def paraboloid(x): return x[0] ** 2 + 2 * x[1] ** 2
    >>> print(hessian(paraboloid, torch.tensor([1, 1])).round())
    [[2. 0.]
     [0. 4.]]

    :param function: function which depends on n variables from x
    :param x0: n - dimensional array
    :param delta_x: precision of two-point formula above (delta_x = h)
    :return: the hessian of function

    .. note::
        If we make delta_x :math:`\\leq` 1e-4 hessian returns matrix with large error rate

    """
    delta_x = max(delta_x, 1e-4)  # Check note
    x0: torch.Tensor = x0.flatten().double()
    hes = torch.zeros(x0.shape[0], x0.shape[0], dtype=torch.float64)

    for i in range(len(x0)):
        delta_i = torch.zeros_like(x0, dtype=torch.float64)
        delta_i[i] += delta_x

        def partial_i(x: torch.Tensor) -> torch.Tensor:
            return (function(x + delta_i) - function(x - delta_i)) / (2 * delta_x)

        hes[i, :] = gradient(partial_i, x0, delta_x)

    return hes
