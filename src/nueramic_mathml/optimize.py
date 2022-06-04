# Optimization algorithms
from __future__ import annotations
from typing import Tuple, Any, Callable, Literal
from .support import HistoryGSS, HistorySPI, HistoryBFGS


def golden_section_search(function: Callable[[float, Any], float],
                          bounds: Tuple[float, float],
                          epsilon: float = 1e-5,
                          type_optimization: Literal['min', 'max'] = 'min',
                          max_iter: int = 500,
                          verbose: bool = False,
                          keep_history: bool = False,
                          **kwargs) -> Tuple[float, HistoryGSS]:
    """
    Returns the optimal point and history using the Golden Section search [1]_

    :math:`\\rule{11cm}{0.7pt}`
    :math:`\\textbf{Constant: } \\displaystyle \\varphi = \\frac{(1 + \\sqrt{5})}{2}`

    :math:`\\textbf{Input: } f(x) - \\text{ function }; a, b - \\text{ left and right bounds };
    \\varepsilon - \\text{ precision }`
    :math:`\\rule{11cm}{0.3pt}`

    :math:`\\text{while } |a - b| > \\varepsilon: \\\\`
    :math:`\\qquad \\displaystyle x_1 = b - \\frac{b - a}{\\varphi} \\\\`
    :math:`\\qquad \\displaystyle x_2 = a + \\frac{b - a}{\\varphi} \\\\`
    :math:`\\qquad \\text{if }  f(x_1) > f(x_2): \\\\`
    :math:`\\qquad \\qquad a = x_1 \\\\`
    :math:`\\qquad \\text{else}: \\\\`
    :math:`\\qquad \\qquad b = x_2`

    :math:`\\rule{11cm}{0.7pt}`

    .. note::
        If optimization fails golden_section_search will return the last point

    Code example:

    .. code-block:: python3

        >>> def func(x): return 2.71828 ** (3 * x) + 5 * 2.71828 ** (-2 * x)
        >>> point, data = golden_section_search(func, (-10, 10), type_optimization='min', keep_history=True)

    .. rubric:: References

    ..  [1] Press, William H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
        Numerical Recipes with Source Code CD-ROM 3rd Edition: The Art of Scientific Computing (3rd ed.).
        Cambridge University Press.

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
                               'f_value': [function((a + b) / 2, **kwargs)],
                               'left_point': [a],
                               'right_point': [b]}

    else:
        history: HistoryGSS = {'iteration': [], 'middle_point': [], 'f_value': [], 'left_point': [], 'right_point': []}

    if verbose:
        print(f'Iteration: {0} \\t|\\t point = {(a + b) / 2 :0.3f} '
              f'\\t|\\t f(point) = {function((a + b) / 2, **kwargs): 0.3f}')

    try:
        for i in range(1, max_iter):
            x1: float = b - (b - a) / phi
            x2: float = a + (b - a) / phi

            if type_optimization == 'min':
                if function(x1, **kwargs) > function(x2, **kwargs):
                    a = x1
                else:
                    b = x2
            else:
                if function(x1, **kwargs) < function(x2, **kwargs):
                    a = x1
                else:
                    b = x2

            middle_point: float = (a + b) / 2
            if verbose:
                print(f'Iteration: {i} \\t|\\t point = {middle_point :0.3f} '
                      f'\\t|\\t f(point) = {function(middle_point, **kwargs): 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['middle_point'].append(middle_point)
                history['f_value'].append(function(middle_point, **kwargs))
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
