# Multi dimensional optimization algorithms for function optimization
from typing import Callable, Tuple, Any

import numpy
import torch

from .calculus import gradient
from .one_optimize import brent
from .support import HistoryBFGS, HistoryGD, update_history_gd, HiddenPrints


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

        with HiddenPrints():
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


def gd_constant_step(function: Callable[[torch.Tensor], float],
                     x0: torch.Tensor,
                     epsilon: float = 1e-5,
                     gamma: float = 0.1,
                     max_iter: int = 500,
                     verbose: bool = False,
                     keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Algorithm with constant step. Documentation: paragraph 2.2.2, page 3.
    The gradient of the function shows us the direction of increasing the function.
    The idea is to move in the opposite direction to x_{k + 1} where f(x_{k + 1}) < f(x_{k}).
    But, if we add a gradient to x_{k} without changes, our method will often diverge.
    So we need to add a gradient with some weight gamma.

    Code example::
        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_constant_step(func, x_0)
        >>> print(solution[0])
        {'point': array([1.91561942e-06, 3.83123887e-06]), 'f_value': 1.834798903191018e-11}

    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param gamma: gradient step
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """
    x_k = x0.double().flatten()
    grad_k = gradient(function, x_k)
    func_k = function(x_k)
    round_precision = -int(numpy.log10(epsilon))  # variable to determine the rounding accuracy

    # if keep_history=True, we will save history. here is initial step
    if keep_history:
        history: HistoryGD = {
            'iteration': [0],
            'f_value': [func_k],
            'f_grad_norm': [torch.sum(grad_k ** 2) ** 0.5],
            'x': [x_k]
        }

    else:
        history: HistoryGD = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}

    # if verbose=True, print the result on each iteration
    if verbose:
        print(f'Iteration: {0} \t|\t '
              f'point = {torch.round(x_k, decimals=round_precision)} \t|\t '
              f'f(point) = {round(func_k, round_precision)}')
    try:
        for i in range(max_iter - 1):

            if torch.sum(grad_k ** 2) ** 0.5 < epsilon:  # comparing of norm 2 with optimization accuracy
                history['message'] = 'Optimization terminated successfully. code 0'
                break
            else:
                x_k = x_k - gamma * grad_k  # updating the point for next iter and repeat
                grad_k = gradient(function, x_k)
                func_k = function(x_k)

            # again, if keep_history=True add the result of the iter
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, torch.sum(grad_k ** 2) ** 0.5, x_k])

            # again, if verbose=True, print the result of the iter
            if verbose:
                print(f'Iteration: {i + 1} \t|\t '
                      f'point = {torch.round(x_k, decimals=round_precision)} \t|\t '
                      f'f(point) = {numpy.round(func_k, decimals=round_precision)}')

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def gd_frac_step(function: Callable[[torch.Tensor], float],
                 x0: torch.Tensor,
                 epsilon: float = 1e-5,
                 gamma: float = 0.1,
                 delta: float = 0.1,
                 lambda0: float = 0.1,
                 max_iter: int = 500,
                 verbose: bool = False,
                 keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Algorithm with fractional step. Documentation: paragraph 2.2.3, page 4
    Requirements: 0 < lambda0 < 1 is the step multiplier, 0 < delta < 1.

    Code example::

        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_frac_step(func, x_0)
        >>> print(solution[0])
        {'point': array([1.91561942e-06, 3.83123887e-06]), 'f_value': 1.834798903191018e-11}

    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param gamma: gradient step
    :param delta: value of the crushing parameter
    :param lambda0: initial step
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """

    x_k = x0.double().flatten()
    func_k = function(x0)
    grad_k = gradient(function, x_k)
    round_precision = -int(numpy.log10(epsilon))  # variable to determine the rounding accuracy

    # if keep_history=True, we will save history. here is initial step
    if keep_history:
        history: HistoryGD = {
            'iteration': [0],
            'f_value': [func_k],
            'f_grad_norm': [sum(grad_k ** 2) ** 0.5],
            'x': [x_k]
        }

    else:
        history: HistoryGD = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}

    # if verbose=True, print the result on each iteration
    if verbose:
        print(f'Iteration: {0} \t|\t '
              f'point = {torch.round(x_k, decimals=round_precision)} \t|\t '
              f'f(point) = {round(func_k, round_precision)}')

    try:
        for i in range(max_iter - 1):

            # point for first comparison, first gradient step
            t = x_k - gamma * grad_k
            func_t = function(t)

            # will divide the gradient step until condition is met
            while not func_t - func_k <= - gamma * delta * sum(grad_k ** 2):
                gamma = gamma * lambda0
                t = x_k - gamma * grad_k
                func_t = function(t)

            x_k = t
            func_k = func_t
            grad_k = gradient(function, x_k)

            # again, if keep_history=True add the result of the iter
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, sum(grad_k ** 2) ** 0.5, x_k])

            # again, if verbose=True, print the result of the iter
            if verbose:
                print(f'Iteration: {i + 1} \t|\t '
                      f'point = {torch.round(x_k, decimals=round_precision)} \t|\t '
                      f'f(point) = {round(func_k, round_precision)}')

            #  comparing of norm 2 with optimization accuracy
            if torch.sum(grad_k ** 2) ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break
        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def gd_optimal_step(function: Callable[[torch.Tensor], float],
                    x0: torch.Tensor,
                    epsilon: float = 1e-5,
                    max_iter: int = 500,
                    verbose: bool = False,
                    keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Algorithm with optimal step. Documentation: paragraph 2.2.4, page 5
    The idea is to choose a gamma that minimizes the function in the direction f'(x_k)

    Code example::

        >>> def func(x): return -torch.e ** (- x[0] ** 2 - x[1] ** 2)
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_optimal_step(func,x_0)
        >>> print(solution[0])
        {'point': array([9.21321369e-08, 1.84015366e-07]), 'f_value': -0.9999999999999577}


    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.
    """

    x_k = x0.double().flatten()
    func_k = function(x_k)
    grad_k = gradient(function, x_k)
    round_precision = -int(numpy.log10(epsilon))  # variable to determine the rounding accuracy

    # if keep_history=True, we will save history. here is initial step
    if keep_history:
        history: HistoryGD = {
            'iteration': [0],
            'f_value': [func_k],
            'f_grad_norm': [sum(grad_k ** 2) ** 0.5],
            'x': [x_k]
        }

    else:
        history: HistoryGD = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}

    # if verbose=True, print the result on each iteration
    if verbose:
        print(f'Iteration: {0} \t|\t '
              f'point = {numpy.round(x_k, round_precision)} \t|\t '
              f'f(point) = {round(func_k, round_precision)}')

    try:
        for i in range(max_iter - 1):

            with HiddenPrints():  # hiding the prints of the results of the brent algorithm
                gamma = brent(lambda gam: function(x_k - gam * grad_k), (0, 1))[0]

            x_k = x_k - gamma * grad_k
            grad_k = gradient(function, x_k)

            # again, if keep_history=True add the result of the iter
            if keep_history:
                func_k = function(x_k)
                history = update_history_gd(history, values=[i + 1, func_k, sum(grad_k ** 2) ** 0.5, x_k])

            # again, if verbose=True, print the result of the iter
            if verbose:
                func_k = function(x_k)
                print(f'Iteration: {i + 1} \t|\t '
                      f'point = {torch.round(x_k, decimals=round_precision)} \t|\t '
                      f'f(point) = {round(func_k, round_precision)}')

            # comparing of norm 2 with optimization accuracy
            if sum(grad_k ** 2) ** 0.5 < float(epsilon):
                history['message'] = 'Optimization terminated successfully. code 0'
                break

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history
