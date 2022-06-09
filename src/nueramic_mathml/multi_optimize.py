# Multi dimensional optimization algorithms for function optimization
from __future__ import annotations

from typing import Callable, Tuple, Sequence

import numpy
import torch

from .calculus import gradient, hessian, jacobian
from .one_optimize import brent
from .support import HistoryBFGS, HistoryGD, update_history_gd, HiddenPrints, print_verbose


def initialize(function: Callable[[torch.Tensor], torch.Tensor],
               x0: torch.Tensor,
               epsilon: float = 1e-5,
               keep_history: bool = False) -> Tuple:
    """
    Returns initial x_k with double dtype, func_k, grad_k, round_precision, history

    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param keep_history: flag of return history
    :return: tuple with x_k, func_k, grad_k, history, round_precision

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
            'f_grad_norm': [(grad_k ** 2).sum() ** 0.5],
            'x': [x_k],
            'message': ''
        }

    else:
        history: HistoryGD = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': [], 'message': ''}

    return x_k, func_k, grad_k, history, round_precision


def bfgs(function: Callable[[torch.Tensor], torch.Tensor],
         x0: torch.Tensor,
         tolerance: float = 1e-8,
         max_iter: int = 500,
         verbose: bool = False,
         keep_history: bool = False) -> Tuple[torch.Tensor, HistoryBFGS]:
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

    # initialization
    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, tolerance, keep_history)
    h_k = torch.eye(x_k.shape[0], dtype=torch.float64) * tolerance ** 0.5
    grad_k = grad_k.reshape(-1, 1)
    x_k = x_k.reshape(-1, 1)

    # first verbose
    print_verbose(x_k, func_k, verbose, 0, round_precision)

    try:
        for i in range(max_iter):

            # stop criterion
            if (grad_k ** 2).sum() ** 0.5 < tolerance:
                print('Searching finished. Successfully. code 0')
                return x_k.reshape(-1), history

            p_k = -h_k @ grad_k

            with HiddenPrints():
                def optimization_f(alpha: float | torch.Tensor) -> float | torch.Tensor:
                    return function(x_k + alpha * p_k)

                alpha_k = brent(optimization_f, (0, 1))[0]

            # step
            x_k_plus1 = x_k + alpha_k * p_k
            grad_f_k_plus1 = gradient(function, x_k_plus1).reshape(-1, 1)
            s_k = x_k_plus1 - x_k
            y_k = grad_f_k_plus1 - grad_k

            h_k = calc_h_new(h_k, s_k, y_k)
            grad_k = grad_f_k_plus1
            x_k = x_k_plus1
            func_k = function(x_k)

            # check divergence
            if torch.isnan(x_k).any():
                print(f'The method has diverged. code 2')
                return x_k.flatten(), history

            # verbose
            print_verbose(x_k, func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, (grad_k ** 2).sum() ** 0.5, x_k])

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    print('Searching finished. Max iterations have been reached. code 1')
    return x_k.flatten(), history


def calc_h_new(h: torch.Tensor,
               s: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    """
    Calculates a new approximation of the inverse Hessian matrix

    :param h: The previous approximation of the H matrix
    :param s: the difference x_{i+1} - x_{i}
    :param y: the difference f'_{i+1} - f'_{i}
    :return: The new approximation of inverse Hessian matrix
    """

    ro = 1 / (y.T @ s)
    i = torch.eye(h.shape[0]).double()
    h_new = (i - ro * s @ y.T) @ h @ (i - ro * s @ y.T) + ro * s @ s.T

    return h_new


def gd_constant(function: Callable[[torch.Tensor], torch.Tensor],
                x0: torch.Tensor,
                epsilon: float = 1e-5,
                gamma: float = 0.1,
                max_iter: int = 500,
                verbose: bool = False,
                keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Algorithm with constant step. Documentation: paragraph 2.2.2, page 3.
    The gradient of the function shows us the direction of increasing the function.
    The idea is to move in the opposite direction to x_{i + 1} where f(x_{i + 1}) < f(x_{i}).
    But, if we add a gradient to x_{i} without changes, our method will often diverge.
    So we need to add a gradient with some weight gamma.

    Code example::
        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_constant(func, x_0)
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

    # initialization
    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)

    # first verbose
    print_verbose(x_k, func_k, verbose, 0, round_precision)

    try:
        for i in range(max_iter - 1):

            if torch.sum(grad_k ** 2) ** 0.5 < epsilon:  # comparing of norm 2 with optimization accuracy
                history['message'] = 'Optimization terminated successfully. code 0'
                break
            else:
                x_k = x_k - gamma * grad_k  # updating the point for next iter and repeat
                grad_k = gradient(function, x_k)
                func_k = function(x_k)

            # verbose
            print_verbose(x_k, func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, (grad_k ** 2).sum() ** 0.5, x_k])

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def gd_frac(function: Callable[[torch.Tensor], torch.Tensor],
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
        >>> solution = gd_frac(func, x_0)
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

    # initialization
    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)

    # first verbose
    print_verbose(x_k, func_k, verbose, 0, round_precision)

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

            #  comparing of norm 2 with optimization accuracy
            if torch.sum(grad_k ** 2) ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break

            # verbose
            print_verbose(x_k, func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, (grad_k ** 2).sum() ** 0.5, x_k])

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def gd_optimal(function: Callable[[torch.Tensor], torch.Tensor],
               x0: torch.Tensor,
               epsilon: float = 1e-5,
               max_iter: int = 500,
               verbose: bool = False,
               keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Algorithm with optimal step. Documentation: paragraph 2.2.4, page 5
    The idea is to choose a gamma that minimizes the function in the direction f'(x_k)

    Code example::

        >>> def func(x): return -torch.exp(- x[0] ** 2 - x[1] ** 2)
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_optimal(func, x_0)
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

    # initialization
    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)

    # first verbose
    print_verbose(x_k, func_k, verbose, 0, round_precision)

    try:
        for i in range(max_iter - 1):

            # comparing of norm 2 with optimization accuracy
            if (grad_k ** 2).sum() ** 0.5 < float(epsilon):
                history['message'] = 'Optimization terminated successfully. code 0'
                break

            with HiddenPrints():  # hiding the prints of the results of the brent algorithm
                def optimization_f(gam: float | torch.Tensor) -> float | torch.Tensor:
                    return function(x_k - gam * grad_k)

                gamma = brent(optimization_f, (0, 1))[0]

            x_k = x_k - gamma * grad_k
            grad_k = gradient(function, x_k)

            # verbose
            print_verbose(x_k, func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, (grad_k ** 2).sum() ** 0.5, x_k])

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def nonlinear_cgm(function: Callable[[torch.Tensor], torch.Tensor],
                  x0: torch.Tensor,
                  epsilon: float = 1e-5,
                  max_iter: int = 500,
                  verbose: bool = False,
                  keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Paragraph 2.4.1 page 6
    Algorithm works when the function is approximately quadratic near the minimum, which is the case when the
    function is twice differentiable at the minimum and the second derivative is non-singular there.


    Code example::
        >>> def func(x): return 10 * x[0] ** 2 + x[1] ** 2 / 5
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = nonlinear_cgm(func, x_0)
        >>> print(solution[0])
        {'point': array([-1.70693616e-07,  2.90227591e-06]), 'f_value': 1.9760041961386155e-12}

    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """

    # initialization
    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)
    p_k = grad_k

    # first verbose
    print_verbose(x_k, func_k, verbose, 0, round_precision)

    try:
        for i in range(max_iter - 1):

            if (grad_k ** 2).sum() ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break
            else:
                with HiddenPrints():
                    def optimization_f(gam: float | torch.Tensor) -> float | torch.Tensor:
                        return function(x_k - gam * grad_k)

                    gamma = brent(optimization_f, (0, 1))[0]

                x_k = x_k - gamma * p_k
                grad_k_new = gradient(function, x_k)
                beta_fr = (grad_k_new @ grad_k_new.reshape(-1, 1)) / (grad_k @ grad_k.reshape(-1, 1))
                p_k = grad_k_new + beta_fr * p_k
                grad_k = grad_k_new
                func_k = function(x_k)

                # verbose
                print_verbose(x_k, func_k, verbose, i + 1, round_precision)

                # history
                if keep_history:
                    history = update_history_gd(history, values=[i + 1, func_k, (grad_k ** 2).sum() ** 0.5, x_k])

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def log_barrier_function(function: Callable[[torch.Tensor], torch.Tensor],
                         x0: torch.Tensor,
                         mu: float,
                         inequality_constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]]
                         ) -> torch.Tensor:
    """
    Support function. Compute log-barrier function .. math::

        P(x, \\mu) = f(x) - \\mu \\sum_{i\\in\\mathcal{I}}\\ln c_i(x)

    :param function:
    :param x0: some specific point x
    :param mu:
    :param inequality_constraints: :math:`\\mathcal{I}` is set of inequality functions
    :return:
    """
    m = len(inequality_constraints)  # Amount of inequality constraints
    output_lb = function(x0)
    for j in range(m):
        const_function = inequality_constraints[j](x0)

        if 0 <= const_function < 1e-8:
            output_lb += mu * 0
        elif const_function > 1e-8:
            output_lb -= mu * torch.log(const_function)
        else:
            output_lb += 10**10

    return output_lb


def primal_dual_interior(function: Callable[[torch.Tensor], torch.Tensor],
                         x0: torch.Tensor,
                         inequality_constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]],
                         mu: float = 1e-4,
                         epsilon: float = 1e-12,
                         alpha: float = 1e-1,
                         max_iter: int = 200,
                         verbose: bool = False,
                         keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Returns point and history of minimization

    AIM: minimize function(x) subject to c(x) >= 0; c from inequality_constraints

    :param function:
    :param x0:
    :param inequality_constraints:
    :param mu: is a small positive scalar, sometimes called the "barrier parameter"
    :param epsilon:
    :param alpha:
    :param max_iter:
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return:

    :raises ArithmeticError: if x0 is not in trust region.

    .. rubric:: Reference
    .. https://en.wikipedia.org/wiki/Interior-point_method

    """

    # initialization
    def lb_function(x: torch.Tensor, _mu: float = mu) -> torch.Tensor | float:
        """ log barrier function -- main function. therefore, instead of a function, we use lb_function """
        return log_barrier_function(function, x, _mu, inequality_constraints)

    x_k, func_k, grad_k, history, round_precision = initialize(lb_function, x0, epsilon, keep_history)
    m = len(inequality_constraints)  # Amount of inequality constraints
    n = x_k.shape[0]  # Amount of variables
    p = torch.ones(m + n)  # init of p
    lambdas = torch.rand(m, dtype=torch.float64)

    try:
        function(x0)
        for i in range(m):
            if inequality_constraints[i](x0) < 0:
                raise ArithmeticError

    except ArithmeticError:
        history['message'] = 'Point out of domain'
        return x_k, history

    except torch.linalg.LinAlgError as e:
        history['message'] = f'Optimization failed. Determinant of function hessian is zero. torch: {e}. code 2.'
        return x_k, history

    # first verbose
    print_verbose(x_k, func_k, verbose, 0, round_precision)
    try:
        for i in range(max_iter):

            # terminate condition
            if p[:n].norm(2) < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break

            w = hessian(lb_function, x_k, 1e-4)  # hessian of lb function
            a = jacobian(inequality_constraints, x_k)
            c = torch.diag(torch.tensor([c(x_k) for c in inequality_constraints])).double()

            left_matrix = torch.zeros(n + m, n + m, dtype=torch.float64)  # left matrix equation
            left_matrix[:n, :n] = w
            left_matrix[:n, n:] = -a.T
            left_matrix[n:, :n] = torch.diag(lambdas) @ a
            left_matrix[n:, n:] = c

            right_matrix = torch.zeros(n + m, 1, dtype=torch.float64)  # right matrix
            # - g + A.T @ lambdas
            right_matrix[:n] = - gradient(lb_function, x_k).reshape(-1, 1) + a.T @ lambdas.reshape(-1, 1)
            # mu 1 - c @ lambdas
            right_matrix[n:] = mu * torch.ones(m, 1) - c @ lambdas.reshape(-1, 1)

            try:  # check singularity
                p = torch.linalg.solve(left_matrix, right_matrix.flatten())  # direction

            except torch.linalg.LinAlgError:
                left_matrix[:n, :n] += 1e-2 * torch.eye(n)
                left_matrix[n:, n:] += 1e-4 * torch.eye(m)

                p = torch.linalg.solve(left_matrix, right_matrix.flatten())  # direction

            x_k += alpha * p[:n]
            lambdas += alpha * p[n:]

            # verbose
            if verbose or keep_history:
                func_k = function(x_k)

            print_verbose(x_k, func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, p[:n].norm(2), x_k.clone()])

            mu *= 0.99

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history
