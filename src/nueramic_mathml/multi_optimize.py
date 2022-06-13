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
    Returns a tensor n x 1 with optimal point and history using the BFGS method [1]_

    Broyden–Fletcher–Goldfarb–Shanno algorithm
    The algorithm does not use Wolfe conditions. Instead of wolfe, alg uses the optimal step.


    .. note::
        The algorithm only works for a flat x0, and the functions should depend on a flat array

    .. rubric:: References

    ..  [1] Wright and Nocedal, 'Numerical Optimization', 1999; pp.136-140 BFGS algorithm.

    :math:`\\rule{125mm}{0.2pt} \\\\`

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param x0: start minimization point
    :param tolerance: criterion of stop os l2 norm(grad f) < tolerance
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    .. rubric:: Examples

    .. code-block:: python3

        >>> def func(x): return 10 * x[0] ** 2 + x[1] ** 2 / 5
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = bfgs(func, x_0)
        >>> print(solution[0])
        tensor([3.4372e-14, 1.8208e-14], dtype=torch.float64)
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
                history['message'] = 'Searching finished. Successfully. code 0'
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
                history['message'] = f'The method has diverged. code 2'
                return x_k.flatten(), history

            # verbose
            print_verbose(x_k.flatten(), func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, (grad_k ** 2).sum() ** 0.5, x_k])

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    history['message'] = 'Searching finished. Max iterations have been reached. code 1'
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
    Returns a tensor n x 1 with optimal point and history using
    Algorithm with constant step.
    The gradient of the function shows us the direction of increasing the function.
    The idea is to move in the opposite direction to :math:`\\displaystyle x_{k + 1} \\text{ where }
    f(x_{k + 1}) < f(x_{k}) \\text{ .}`

    But, if we add a gradient to :math:`\\displaystyle x_{k}` without changes, our method will often diverge.
    So we need to add a gradient with some weight :math:`\\displaystyle \\gamma \\text{ .}\\\\`


    :param function: callable that depends on the first positional argument
    :param x0: Torch tensor which is initial approximation
    :param epsilon: optimization accuracy
    :param gamma: gradient step
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    .. rubric:: Examples

    .. code-block:: python3

        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_constant(func, x_0)
        >>> print(solution[0])
        tensor([1.9156e-06, 3.8312e-06], dtype=torch.float64)
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
    Returns a tensor n x 1 with optimal point and history using
    Algorithm with fractional step.

    Requirements: :math:`\\ 0 < \\lambda_0 < 1`  is the step multiplier, :math:`0 < \\delta < 1` influence on step size.

    :param function: callable that depends on the first positional argument
    :param x0: Torch tensor which is initial approximation
    :param epsilon: optimization accuracy
    :param gamma: gradient step
    :param delta: value of the crushing parameter
    :param lambda0: initial step
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.



    .. rubric:: Examples

    .. code-block:: python3

        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_frac(func, x_0)
        >>> print(solution[0])
        tensor([1.9156e-06, 3.8312e-06], dtype=torch.float64)

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
    Returns a tensor n x 1 with optimal point and history using
    Algorithm with optimal step.
    The idea is to choose a gamma that minimizes the function in the direction :math:`\\ f'(x_k)`



    :param function: callable that depends on the first positional argument
    :param x0: Torch tensor which is initial approximation
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    .. rubric:: Examples

    .. code-block:: python3

        >>> def func(x): return -torch.exp(- x[0] ** 2 - x[1] ** 2)
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = gd_optimal(func, x_0)
        >>> print(solution[0])
        tensor([9.2070e-08, 1.8405e-07], dtype=torch.float64)
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


def nonlinear_cgm(function: Callable[[torch.Tensor], torch.Tensor],
                  x0: torch.Tensor,
                  epsilon: float = 1e-5,
                  max_iter: int = 500,
                  verbose: bool = False,
                  keep_history: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Returns a tensor n x 1 with optimal point and history.
    Algorithm works when the function is approximately quadratic near the minimum, which is the case when the
    function is twice differentiable at the minimum and the second derivative is non-singular there [1]_

    .. rubric:: References

    ..  [1] Nocedal, J., &amp; Wright, S. J. (2006). 5.2 NONLINEAR CONJUGATE GRADIENT METHOD
        In Numerical optimization (pp. 121). essay, Springer.

    :param function: callable that depends on the first positional argument
    :param x0: Torch tensor which is initial approximation
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.


    .. rubric:: Examples

    .. code-block:: python3

        >>> def func(x): return 10 * x[0] ** 2 + x[1] ** 2 / 5
        >>> x_0 = torch.tensor([1, 2])
        >>> solution = nonlinear_cgm(func, x_0)
        >>> print(solution[0])
        tensor([6.9846e+25, 4.2454e+26], dtype=torch.float64)
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
    Support function. Compute log-barrier function

    .. math::

        P(x, \\mu) = f(x) - \\mu \\sum_{i\\in\\mathcal{I}}\\ln c_i(x)

    :param function: callable that depends on the first positional argument
    :param x0: some specific point x(Torch tensor)
    :param mu: parameter weighing constraints
    :param inequality_constraints: :math:`\\mathcal{I}` is set of inequality functions
    :return: tuple with point and history.
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
            output_lb += 10 ** 10

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
    Returns point and history of minimization. [1]_

    AIM: minimize :math:`\\ f(x)` subject to :math:`\\ c(x) \\geqslant 0`; c from inequality_constraints

    :math:`\\ B(x,\\mu) = f(x) - \\mu \\sum_{i=1}^m \\log(c_i(x)) \\rightarrow \\min`

    Here :math:`\\mu` is a small positive scalar, :math:`\\mu \\rightarrow 0`

    :param function: callable that depends on the first positional argument
    :param x0: some specific point x(Torch tensor)
    :param inequality_constraints: :math:`\\mathcal{I}` is set of inequality functions
    :param mu: is a small positive scalar, sometimes called the "barrier parameter"
    :param epsilon: optimization accuracy
    :param alpha: step length
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return:  tuple with point and history.

    :raises ArithmeticError: if x0 is not in trust region.

    .. rubric:: Reference

    .. [1] https://en.wikipedia.org/wiki/Interior-point_method

    .. rubric:: Examples

    .. code-block:: python3


        >>> primal_dual_interior(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2, torch.tensor([0.9, 0.1]),
        >>>                    [lambda x: x[0], lambda x: 1 - x[0], lambda x: x[1], lambda x: 1 - x[1]])[0]
        tensor([1.9910e-04, 5.0000e-01], dtype=torch.float64)
    """

    # initialization
    def lb_function(x: torch.Tensor, _mu: float = mu) -> torch.Tensor | float:
        """ log barrier function -- main function. therefore, instead of a function, we use lb_function """
        return log_barrier_function(function, x, _mu, inequality_constraints)

    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)
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


def log_barrier_solver(function: Callable[[torch.Tensor], torch.Tensor],
                       x0: torch.Tensor,
                       inequality_constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]],
                       epsilon: float = 1e-5,
                       max_iter: int = 1000,
                       keep_history: bool = False,
                       verbose: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Returns optimal point of optimization with inequality constraints by Log Barrier method [1]_




    .. rubric:: References

    ..  [1] Nocedal, J., &amp; Wright, S. J. (2006). 19.6 THE PRIMAL LOG-BARRIER METHOD.
            In Numerical optimization (pp. 583–584). essay, Springer.

    :math:`\\rule{125mm}{0.2pt} \\\\`

    :param function: callable that depends on the first positional argument
    :param x0: some specific point x(Torch tensor)
    :param epsilon: optimization accuracy
    :param inequality_constraints: :math:`\\mathcal{I}` is set of inequality functions
    :param max_iter: maximum number of iterations
    :param keep_history: flag of return history
    :param verbose: flag of printing iteration logs
    :return: tuple with point and history.


    .. rubric:: Examples

    Example for :math:`f(x, y) = (x + 0.5)^2 + (y - 0.5)^2, \\quad 0 \\le x \\le 1, 0 \\le y \\le 1`

    .. code-block:: python3

        >>> log_barrier_solver(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2, torch.tensor([0.9, 0.1]),
        >>>                    [lambda x: x[0], lambda x: 1 - x[0], lambda x: x[1], lambda x: 1 - x[1]])
        tensor([0.0032, 0.5000], dtype=torch.float64)

    """
    m = len(inequality_constraints)  # Amount of inequality constraints
    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)

    try:
        function(x0)
        for i in range(m):
            if inequality_constraints[i](x0) < 0:
                raise ArithmeticError

    except ArithmeticError:
        history['message'] = 'Point out of domain'
        return x_k, history

    tau = 1  # The tau sequence will be geometric

    try:
        for i in range(max_iter):
            mu_k = tau ** 0.5
            x_k, history_step = gd_frac(lambda x: log_barrier_function(function, x, mu_k, inequality_constraints),
                                        x_k, gamma=mu_k, epsilon=tau, keep_history=True, max_iter=5)

            tau *= 0.9
            if tau <= epsilon:
                break

            # verbose
            if verbose or keep_history:
                func_k = function(x_k)
                grad_k = gradient(function, x_k)

            print_verbose(x_k, func_k, verbose, i + 1, round_precision)

            # history
            if keep_history:
                history = update_history_gd(history, values=[i + 1, func_k, grad_k.norm(2), x_k.clone()])

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return x_k, history


def constrained_lagrangian_solver(function: Callable[[float | torch.Tensor], torch.Tensor],
                                  x0: torch.Tensor,
                                  constraints: Sequence[Callable[[float | torch.Tensor], torch.Tensor]],
                                  x_bounds: Sequence[Tuple[float, float]] | None | torch.Tensor = None,
                                  epsilon: float = 1e-4,
                                  max_iter: int = 250,
                                  keep_history: bool = False,
                                  verbose: bool = False) -> Tuple[torch.Tensor, HistoryGD]:
    """
    Returns a tensor n x 1 with optimal point and history of minimization by newton_eq_const.
    Alias of ''Newton’s method under equality constrains'' [1]_

    Example for :math:`f(x, y) = (x + 0.5)^2 + (y - 0.5)^2, \\quad x = 1`

    .. rubric:: References

    ..  [1] Nocedal, J., &amp; Wright, S. J. (2006). 17.4 PRACTICAL AUGMENTED LAGRANGIAN METHODS.
        In Numerical optimization (pp. 519–521). essay, Springer.

    :math:`\\rule{125mm}{0.2pt} \\\\`

    :param function: callable that depends on the first positional argument
    :param x0: some specific point x(Torch tensor)
    :param constraints: list of equality constraints
    :param x_bounds: bounds on x. e.g. 0 <= x[i] <= 1, then x_bounds[i] = (0, 1)
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param keep_history: flag of return history
    :param verbose: flag of printing iteration logs
    :return: tuple with point and history.

    .. rubric:: Examples

    .. code-block:: python3

        >>> constrained_lagrangian_solver(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2,
        >>>                              torch.tensor([0.1, 0.1]),[lambda x: x[0] - 1]))
        tensor([1.0540, 0.5000], dtype=torch.float64)
    """
    m = len(constraints)
    if x_bounds is None:
        x_bounds = torch.zeros(len(x0), 2).double()
        for i in range(len(x0)):
            x_bounds[i, :] = torch.tensor([-torch.inf, torch.inf])

    elif isinstance(x0, torch.Tensor):
        assert x0.shape[0] == x_bounds.shape[0], 'the boundaries should be n x 2 tensor'
        assert len(x_bounds.shape) == 2, 'the boundaries should be n x 2 tensor'
        assert x_bounds.shape[1] == 2, 'the boundaries should be n x 2 tensor'

    else:
        assert len(x_bounds) == x0.shape[0], 'the boundaries should be for each variable'

    if not isinstance(x0, torch.Tensor):

        _x_bounds = torch.zeros(x0.shape[0], 2).double()
        for i, const in enumerate(x_bounds):
            _x_bounds[i, :] = const
        x_bounds = _x_bounds

    def c(x: torch.Tensor) -> torch.Tensor:
        """Returns vector of constraints at specific x"""
        _c = torch.zeros(m).double()
        for j in range(m):
            _c[j] = constraints[j](x)
        return _c

    def lagrangian_a(x: torch.Tensor, lam: torch.Tensor, mu: float) -> torch.Tensor:
        """
        Returns :math:`\\mathcal{L}_a`
        Nocedal, J., &amp; Wright, S. J. (2006). Numerical optimization (p. 520)
        """

        output = function(x)

        for j in range(m):
            output += -lam[j] * constraints[j](x) + mu / 2 * constraints[j](x) ** 2

        return output

    def p_function(g: torch.Tensor, u_l_bounds: torch.Tensor):
        """
        P(g, l, u) is the projection of the vector g :math:`\\in` IRn onto the rectangular box :math:`[l, u]`
        Nocedal, J., &amp; Wright, S. J. (2006). Numerical optimization (p. 520)
        """

        # for j in range(len(g)):
        #     if g[j] >= u_l_bounds[j][1]:
        #         g[j] = u_l_bounds[j][1]
        #     elif g[j] <= u_l_bounds[j][0]:
        #         g[j] = u_l_bounds[j][0]

        # replaced by

        g = torch.minimum(torch.maximum(g, u_l_bounds[:, 0]), u_l_bounds[:, 1])

        return g

    x_k, func_k, grad_k, history, round_precision = initialize(function, x0, epsilon, keep_history)

    try:
        function(x0)
        for i in range(m):
            constraints[i](x0)

    except ArithmeticError:
        history['message'] = 'Point out of domain'
        return x_k, history

    lambdas_k = torch.rand(m)
    eta = epsilon  # Main tolerance for constraints
    omega = eta  # Main tolerance for lagrange function
    mu_k = 10
    omega_k = 1 / mu_k
    eta_k = 1 / mu_k ** 0.1

    for i in range(max_iter):

        def local_min_function(x):
            grad_lagrangian = x - gradient(lambda y: lagrangian_a(y, lambdas_k, mu_k), x)
            p = p_function(grad_lagrangian, x_bounds)
            return (x - p).norm(2)

        x_k = nonlinear_cgm(local_min_function, x_k, epsilon=omega_k, max_iter=4)[0]

        if verbose or keep_history:
            func_k = function(x_k)
            grad_k = gradient(function, x_k)

        print_verbose(x_k, func_k, verbose, i + 1, round_precision)

        # history
        if keep_history:
            history = update_history_gd(history, values=[i + 1, func_k, grad_k.norm(2), x_k.clone()])

        c_k = c(x_k)
        if c_k.norm(2) <= eta_k:
            # test for convergence
            if c_k.norm(2) <= eta and local_min_function(x_k) <= omega:
                break

            # update multipliers, tighten tolerances
            lambdas_k = lambdas_k - mu_k * c_k

        else:
            # increase penalty parameter, tighten tolerances
            lambdas_k = lambdas_k - mu_k * c_k
            mu_k = mu_k * 100
            eta_k = 1 / mu_k ** 0.1
            omega = 1 / mu_k

    return x_k, history
