# One dimensional optimization algorithms for function optimization
from __future__ import annotations

import sys
from typing import Tuple, Callable

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
import torch

from .support import HistoryGSS, HistorySPI, HistoryBrent, update_history_brent


def golden_section_search(function: Callable[[float | torch.Tensor], float],
                          bounds: Tuple[float, float],
                          epsilon: float = 1e-5,
                          type_optimization: Literal['min', 'max'] = 'min',
                          max_iter: int = 500,
                          verbose: bool = False,
                          keep_history: bool = False) -> Tuple[float | torch.Tensor, HistoryGSS]:
    """
    Returns the optimal point and history using the Golden Section search [2]_

    :math:`\\rule{125mm}{0.7pt} \\\\`
    :math:`\\textbf{Constant: } \\displaystyle \\varphi = \\frac{(1 + \\sqrt{5})}{2} \\\\`
    :math:`\\textbf{Input: } f(x) - \\text{ function }; a, b - \\text{ left and right bounds };
    \\varepsilon - \\text{ precision } \\\\`
    :math:`\\rule{125mm}{0.3pt}\\\\`

    :math:`\\text{while } |a - b| > \\varepsilon: \\\\`
    :math:`\\qquad \\displaystyle x_1 = b - \\frac{b - a}{\\varphi} \\\\`
    :math:`\\qquad \\displaystyle x_2 = a + \\frac{b - a}{\\varphi} \\\\`
    :math:`\\qquad \\text{if }  f(x_1) > f(x_2): \\\\`
    :math:`\\qquad \\qquad a = x_1 \\\\`
    :math:`\\qquad \\text{else}: \\\\`
    :math:`\\qquad \\qquad b = x_2 \\\\`
    :math:`\\rule{125mm}{0.3pt}\\\\`
    :math:`\\textbf{Return: } \\displaystyle \\frac{a+b}{2} \\\\`
    :math:`\\rule{125mm}{0.7pt} \\\\`

    .. note::
        If optimization fails golden_section_search will return the last point

    Code example:

    .. code-block:: python3

        >>> def func(x): return 2.71828 ** (3 * x) + 5 * 2.71828 ** (-2 * x)
        >>> point, data = golden_section_search(func, (-10, 10), type_optimization='min', keep_history=True)

    .. rubric:: References

    ..  [2] Press, William H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
        Numerical Recipes with Source Code CD-ROM 3rd Edition: The Art of Scientific Computing (3rd ed.).
        Cambridge University Press. p.492-496

    :math:`\\rule{125mm}{0.2pt} \\\\`

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """
    phi: float = (1 + 5 ** 0.5) / 2

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    a: float = bounds[0]
    b: float = bounds[1]
    if keep_history:
        history: HistoryGSS = {'iteration': [0],
                               'middle_point': [(a + b) / 2],
                               'f_value': [function((a + b) / 2)],
                               'left_point': [a],
                               'right_point': [b]}

    else:
        history: HistoryGSS = {'iteration': [], 'middle_point': [], 'f_value': [], 'left_point': [], 'right_point': []}

    if verbose:
        print(f'Iteration: {0} \\t|\\t point = {(a + b) / 2 :0.3f} '
              f'\\t|\\t f(point) = {function((a + b) / 2): 0.3f}')

    try:
        for i in range(1, max_iter):
            x1: float = b - (b - a) / phi
            x2: float = a + (b - a) / phi

            if type_optimization == 'min':
                if function(x1) > function(x2):
                    a = x1
                else:
                    b = x2
            else:
                if function(x1) < function(x2):
                    a = x1
                else:
                    b = x2

            middle_point: float = (a + b) / 2
            if verbose:
                print(f'Iteration: {i} \\t|\\t point = {middle_point :0.3f} '
                      f'\\t|\\t f(point) = {function(middle_point): 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['middle_point'].append(middle_point)
                history['f_value'].append(function(middle_point))
                history['left_point'].append(a)
                history['right_point'].append(b)

            if abs(x1 - x2) < epsilon:
                print('Searching finished. Successfully. code 0')
                return middle_point, history
        else:
            middle_point = (a + b) / 2
            print('Searching finished. Max iterations have been reached. code 1')
            return middle_point, history

    except Exception as e:
        print('Error with optimization. code 2')
        raise e


def successive_parabolic_interpolation(function: Callable[[float | torch.Tensor], float],
                                       bounds: Tuple[float, float],
                                       epsilon: float = 1e-5,
                                       type_optimization: Literal['min', 'max'] = 'min',
                                       max_iter: int = 500,
                                       verbose: bool = False,
                                       keep_history: bool = False) -> Tuple[float | torch.Tensor, HistorySPI]:
    """
    Returns the optimal point and history using the Successive parabolic interpolation algorithm [3]_

    :math:`\\rule{125mm}{0.7pt} \\\\`

    .. math::
        :label: eq1

        \\displaystyle x_{i+1}=x_{i}+ \\frac{1}{2}\\left[\\frac{\\left(x_{i-1}-x_{i}\\right)^{2}
        \\left(f_{i}-f_{i-2}\\right)+
        \\left(x_{i-2}-x_{i}\\right)^{2}\\left(f_{i-1}-f_{i}\\right)}{\\left(x_{i-1}-x_{i}\\right)
        \\left(f_{i}-f_{i-2}\\right)+\\left(x_{i-2}-x_{i}\\right)\\left(f_{i-1}-f_{i}\\right)}\\right]\\\\


    :math:`\\rule{125mm}{0.3pt}\\\\`

    :math:`\\textbf{Input: } f(x) - \\text{ function}; a, b - \\text{ left and right bounds};
    \\varepsilon - \\text{ precision } \\\\`
    :math:`\\rule{125mm}{0.3pt}\\\\`

    :math:`\\displaystyle x_0 = a, \\ f_0 = f(x_0); \\qquad  x_1 = b, \\ f_1 = f(x_1); \\qquad x_2 = \\displaystyle
    \\frac{a+b}{2}, \\ f_2 = f(x_2)\\\\`
    :math:`\\text{while } |x_{i+1}-x_{i}| \\geq \\varepsilon` or :math:`|f(x_{i+1})-f(x_{i})| \\geq \\varepsilon:\\\\`
    :math:`\\qquad \\displaystyle x_0, x_1, x_2` so that :math:`f_2 \\leq f_1 \\leq f_0\\\\`
    :math:`\\qquad \\displaystyle \\text{Calculate } x_{i + 1} \\text{with the formula }`  :eq:`eq1` :math:`\\\\`

    :math:`\\rule{125mm}{0.3pt}\\\\`
    :math:`\\textbf{Return: } \\displaystyle x_{i+1} \\\\`
    :math:`\\rule{125mm}{0.7pt} \\\\`


    Example:
        >>> def func1(x): return x ** 3 - x ** 2 - x
        >>> successive_parabolic_interpolation(func1, (0, 1.5), verbose=True)
        Iteration: 0	|	x2 = 0.750	|	f(x2) = -0.891
        Iteration: 1	|	x2 = 0.850	|	f(x2) = -0.958
        Iteration: 2	|	x2 = 0.961	|	f(x2) = -0.997
        Iteration: 3	|	x2 = 1.017	|	f(x2) = -0.999
        Iteration: 4	|	x2 = 1.001	|	f(x2) = -1.000
        ...

        >>> def func2(x): return - (x ** 3 - x ** 2 - x)
        >>> successive_parabolic_interpolation(func2, (0, 1.5), type_optimization='max', verbose=True)
        Iteration: 0	|	x2 = 0.750	|	f(x2) = 0.891
        Iteration: 1	|	x2 = 0.850	|	f(x2) =  0.958
        Iteration: 2	|	x2 = 0.961	|	f(x2) =  0.997
        Iteration: 3	|	x2 = 1.017	|	f(x2) =  0.999
        ...

    ..  [3] Press, William H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
        Numerical Recipes with Source Code CD-ROM 3rd Edition: The Art of Scientific Computing (3rd ed.).
        Cambridge University Press. p.496-499

    :math:`\\rule{125mm}{0.2pt} \\\\`

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """
    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    if type_optimization == 'max':
        type_opt_const = -1
    else:
        type_opt_const = 1

    history: HistorySPI = {'iteration': [], 'f_value': [], 'x0': [], 'x1': [], 'x2': []}
    x0, x1 = bounds[0], bounds[1]
    x2 = (x0 + x1) / 2
    f0 = type_opt_const * function(x0)
    f1 = type_opt_const * function(x1)
    f2 = type_opt_const * function(x2)
    f_x: dict = {x0: f0, x1: f1, x2: f2}
    x2, x1, x0 = sorted([x0, x1, x2], key=lambda x: f_x[x])

    if keep_history:
        history['iteration'].append(0)
        history['f_value'].append(type_opt_const * f2)
        history['x0'].append(x0)
        history['x1'].append(x1)
        history['x2'].append(x2)

    if verbose:
        print(f'Iteration: {0}\t|\tx2 = {x2:0.3f}\t|\tf(x2) = {type_opt_const * f2: 0.3f}')

    try:
        for i in range(1, max_iter):
            f0, f1, f2 = f_x[x0], f_x[x1], f_x[x2]
            p = (x1 - x2) ** 2 * (f2 - f0) + (x0 - x2) ** 2 * (f1 - f2)
            q = 2 * ((x1 - x2) * (f2 - f0) + (x0 - x2) * (f1 - f2))

            if p != 0:
                print('Searching finished. Select an another initial state. Numerator is zero. code 2')
                return x2, history
            if q != 0:
                print('Searching finished. Select an another initial state. Denominator is zero. code 2')
                return x2, history

            x_new = x2 + p / q

            if not bounds[0] <= x_new <= bounds[1]:
                print('Searching finished. Out of bounds. code 1')
                return x2, history

            f_new = type_opt_const * function(x_new)
            f_x[x_new] = f_new
            previous_xs = [x0, x1, x2]

            if f_new < f2:
                x0, f0 = x1, f1
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new

            elif f_new < f1:
                x0, f0 = x1, f1
                x1, f1 = x_new, f_new

            elif f_new < f0:
                x0, f0 = x_new, f_new

            if verbose:
                print(f'Iteration: {i}\t|\tx2 = {x2:0.3f}\t|\tf(x2) = {type_opt_const * f2: 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['f_value'].append(type_opt_const * f2)
                history['x0'].append(x0)
                history['x1'].append(x1)
                history['x2'].append(x2)

            # In addition, check the criterion when the points don't change
            change_flag = max(map(lambda x, y: abs(x - y), [x0, x1, x2], previous_xs)) < epsilon
            if abs(x1 - x2) < epsilon and abs(f1 - f2) < epsilon or change_flag:
                print('Searching finished. Successfully. code 0')
                return x2, history

        else:
            print('Searching finished. Max iterations have been reached. code 1')
            return x2, history

    except Exception as e:
        print('Error with optimization. code 2')
        raise e


def brent(function: Callable[[float | torch.Tensor], float],
          bounds: Tuple[float, float],
          epsilon: float = 1e-5,
          type_optimization: Literal['min', 'max'] = 'min',
          max_iter: int = 500,
          verbose: bool = False,
          keep_history: bool = False) -> Tuple[float | torch.Tensor, HistoryBrent]:
    """
    Returns the optimal point and history using the Brent's algorithm [1]_.

    :math:`\\rule{125mm}{0.7pt} \\\\`
    :math:`\\textbf{Input: } f(x) - \\text{ function }; a, b - \\text{ left and right bounds };
    \\varepsilon - \\text{ precision } \\\\`
    :math:`\\rule{125mm}{0.3pt}\\\\`

    :math:`\\displaystyle \\varphi = \\frac{(1 + \\sqrt{5})}{2} \\\\`
    :math:`\\displaystyle x_{least} = a + \\varphi \\cdot (b - a) \\\\`
    :math:`\\displaystyle x_{new} = x_{least} \\\\`
    :math:`\\displaystyle tolerance = \\varepsilon \\cdot | x_{least}| + 10^{-9} \\\\`

    :math:`\\text{while }\\displaystyle |x_{least} - \\frac{a+b}{2}| > 2 \\cdot tolerance - \\frac{b-a}{2} :\\\\`

    :math:`\\qquad  \\text{if }\\displaystyle |x_{new} - x_{least}| > tolerance:\\\\`
    :math:`\\qquad \\qquad \\text{calculate parabolic } remainder \\text{ by formula }` :eq:`eq1` :math:`\\\\`
    :math:`\\qquad \\text{if } \\displaystyle remainder < previous \\ remainder \\ \\& \\
    x_{least} + remainder \\in (a, b):\\\\`

    :math:`\\qquad \\qquad  \\text{use  ``paraboloic" } \\displaystyle remainder\\\\`

    :math:`\\qquad \\text{else:}\\\\`
    :math:`\\qquad \\qquad \\text{make  ``golden"  } \\displaystyle remainder\\\\`
    :math:`\\qquad \\qquad \\text{use ``golden" } \\displaystyle remainder\\\\`
    :math:`\\qquad \\displaystyle x_{new} = x_{least} + remainder\\\\`

    :math:`\\rule{125mm}{0.3pt}\\\\`
    :math:`\\textbf{Return: } \\displaystyle x_{least} \\\\`
    :math:`\\rule{125mm}{0.7pt} \\\\`

    .. rubric:: References

    .. [1] Brent, R. P., Algorithms for Minimization Without Derivatives. Englewood Cliffs, NJ: Prentice-Hall,
        1973 pp.72-80

    :math:`\\rule{125mm}{0.2pt} \\\\`

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    if type_optimization == 'max':
        type_opt_const = -1
    else:
        type_opt_const = 1

    gold_const = (3 - 5 ** 0.5) / 2
    remainder = 0.0  # p / q when we calculate x_new

    # initial values
    a, b = sorted(bounds)
    x_largest = x_middle = x_least = a + gold_const * (b - a)
    f_largest = f_middle = f_least = type_opt_const * function(x_least)
    x_least: float | torch.Tensor

    history: HistoryBrent = {
        'iteration': [],
        'f_least': [],
        'f_middle': [],
        'f_largest': [],
        'x_least': [],
        'x_middle': [],
        'x_largest': [],
        'left_bound': [],
        'right_bound': [],
        'type_step': []
    }

    if keep_history:
        history = update_history_brent(
            history,
            [0, f_least, f_middle, f_largest, x_least, x_middle, x_largest, a, b, 'initial']
        )

    if verbose:
        print(f'iteration 0\tx = {x_least:0.6f},\tf(x) = {f_least:0.6f}\ttype : initial')

    for i in range(1, max_iter + 1):
        middle_point = (a + b) / 2
        tolerance = epsilon * abs(x_least) + 1e-9  # f is never evaluated at two points closer together than tolerance

        # Check stopping criterion
        if abs(x_least - middle_point) > 2 * tolerance - (b - a) / 2:
            p = q = previous_remainder = 0
            if abs(remainder) > tolerance:

                # fit parabola
                p = ((x_least - x_largest) ** 2 * (f_least - f_middle) -
                     (x_least - x_middle) ** 2 * (f_least - f_largest))

                q = 2 * ((x_least - x_largest) * (f_least - f_middle) -
                         (x_least - x_middle) * (f_least - f_largest))

                # change q sign to positive
                if q > 0:
                    p = -p
                else:
                    q = -q
                # r stores the previous value of remainder
                previous_remainder = remainder

            # Check conditions for parabolic step:
            # tol - x_new must not be close to x_least, so we check the step
            # previous_remainder - the value of p / q at the second-last cycle
            # |previous_remainder| > tol - is checked above
            # q != 0 - includes in next conditions
            # x_least + p / q in (a, b). New point in interval
            # p / q < previous(p / q) / 2. Control the divergence

            if abs(p) < 0.5 * abs(q * previous_remainder) and a * q < x_least * q + p < b * q:
                remainder = p / q
                x_new = x_least + remainder
                name_step = 'parabolic'

                # Check that f not be evaluated too close to a or b
                if x_new - a < 2 * tolerance or b - x_new < 2 * tolerance:
                    if x_least < middle_point:
                        remainder = tolerance
                    else:
                        remainder = -tolerance

            # If conditions above is false we do golden section step
            else:
                name_step = 'golden'
                if x_least < middle_point:
                    remainder = (b - x_least) * gold_const
                else:
                    remainder = (a - x_least) * gold_const

            # Check that f not be evaluated too close to x_least
            if abs(remainder) > tolerance:
                x_new = x_least + remainder
            elif remainder > 0:
                x_new = x_least + tolerance
            else:
                x_new = x_least - tolerance

            f_new = type_opt_const * function(x_new)

            # Update a, b, x_largest, x_middle, x_leas
            if f_new <= f_least:
                if x_new < x_least:
                    b = x_least
                else:
                    a = x_least

                x_largest = x_middle
                f_largest = f_middle

                x_middle = x_least
                f_middle = f_least

                x_least = x_new
                f_least = f_new

            else:
                if x_new < x_least:
                    a = x_new
                else:
                    b = x_new

                if f_new <= f_middle:
                    x_largest = x_middle
                    f_largest = f_middle

                    x_middle = x_new
                    f_middle = f_new

                elif f_new <= f_largest:
                    x_largest = x_new
                    f_largest = f_new

        else:
            print('Searching finished. Successfully. code 0')
            return x_least, history

        if keep_history:
            history = update_history_brent(
                history,
                [i,
                 type_opt_const * f_least,
                 type_opt_const * f_middle,
                 type_opt_const * f_largest,
                 x_least,
                 x_middle,
                 x_largest,
                 a,
                 b,
                 name_step
                 ]
            )

        if verbose:
            print(f'iteration {i}\tx = {x_least:0.6f},\tf(x) = {type_opt_const * f_least:0.6f}\ttype : {name_step}')

    else:
        print('Searching finished. Max iterations have been reached. code 1')
        return x_least, history

# enf of One dimensional part
