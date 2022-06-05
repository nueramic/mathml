from typing import Callable, Tuple, Any

import torch

from .calculus import gradient
from .one_optimize import brent
from .support import HistoryBFGS


def bfgs(function: Callable[[torch.Tensor, Any], float],
         x0: torch.Tensor,
         tolerance: float = 1e-8,
         max_iter: int = 500,
         verbose: bool = False,
         keep_history: bool = False,
         **kwargs) -> Tuple[torch.Tensor, HistoryBFGS]:
    """
    [СЮДА ОФОРМИТЬ РЕФЕРЕНС] Returns a tensor n x 1 with optimal point using the BFGS method.
    Broyden–Fletcher–Goldfarb–Shanno algorithm
    The algorithm does not use Wolfe conditions. Instead of wolfe, alg uses the optimal step.

    .. note::
        The algorithm only works for a flat x0, and the functions should depend on a flat array

    Wright and Nocedal, 'Numerical Optimization', 1999; pp.136-140 BFGS algorithm.

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param x0: start minimization point
    :param tolerance: criterion of stop os l2 norm(grad f) < tolerance
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.
    """

    x_k = x0.reshape(-1, 1).double()
    h_k = torch.eye(len(x_k)).double() * tolerance ** 0.5
    grad_f_k = gradient(function, x_k).reshape(-1, 1)
    f_k = function(x_k, **kwargs)

    history = {'iteration': [], 'point': [], 'function': []}
    if keep_history:
        history['iteration'].append(0)
        history['point'].append(x_k.reshape(-1))
        history['function'].append(f_k)

    if verbose:
        print(f'iteration: {0} \t x = {torch.round(x_k.reshape(-1), decimals=3)} \t f(x) = {f_k : 0.3f}')

    for k in range(max_iter):

        if sum(grad_f_k ** 2) ** 0.5 < tolerance:
            print('Searching finished. Successfully. code 0')
            return x_k.reshape(-1), history

        p_k = -h_k @ grad_f_k

        alpha_k = brent(lambda alpha: function(x_k + alpha * p_k, **kwargs), (0, 1))[0]

        # if alpha_k is None:
        #     alpha_k = min(tolerance * 10, 0.01)

        x_k_plus1 = x_k + alpha_k * p_k
        grad_f_k_plus1 = gradient(function, x_k_plus1, **kwargs).reshape(-1, 1)
        s_k = x_k_plus1 - x_k
        y_k = grad_f_k_plus1 - grad_f_k

        h_k = calc_h_new(h_k, s_k, y_k)
        grad_f_k = grad_f_k_plus1
        x_k = x_k_plus1
        f_k = function(x_k, **kwargs)

        if torch.isnan(x_k).any():
            print(f'The method has diverged. code 2')
            return x_k.reshape(-1), history

        if verbose:
            print(f'iteration: {k + 1} \t x = {torch.round(x_k.reshape(-1), decimals=3)} \t f(x) = {f_k: 0.3f}')

        if keep_history:
            history['iteration'].append(k + 1)
            history['point'].append(x_k.reshape(-1))
            history['function'].append(f_k)

    print('Searching finished. Max iterations have been reached. code 1')
    return x_k.reshape(-1), history


def calc_h_new(h: torch.Tensor,
               s: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    """
    Calculates a new approximation of the inverse Hessian matrix

    :param h: The previous approximation of the H matrix
    :param s: the difference x_{k+1} - x_{k}
    :param y: the difference f'_{k+1} - f'_{k}
    :return: The new approximation of inverse Hessian matrix
    """

    ro = 1 / (y.T @ s)
    i = torch.eye(h.shape[0]).double()
    h_new = (i - ro * s @ y.T) @ h @ (i - ro * s @ y.T) + ro * s @ s.T

    return h_new
