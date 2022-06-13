from __future__ import annotations

import os
import sys
from typing import Text, Tuple, Sequence, Any

if sys.version_info >= (3, 8):
    from typing import TypedDict, Literal  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal

import torch


class HistoryGSS(TypedDict):
    """
    Class with an optimization history of Golden section search
    """
    iteration: list
    middle_point: list[float]
    f_value: list[float]
    left_point: list[float]
    right_point: list[float]


class HistorySPI(TypedDict):
    """
    Class with an optimization history of Successive parabolic interpolation
    """
    iteration: list[int]
    f_value: list[float]
    x0: list[float]
    x1: list[float]
    x2: list[float]


class HistoryBrent(TypedDict):
    """
    Class with an optimization history of Brant's algorithm
    """
    iteration: list[int]

    f_least: list[float]
    f_middle: list[float]
    f_largest: list[float]

    x_least: list[float]
    x_middle: list[float]
    x_largest: list[float]

    left_bound: list[float]
    right_bound: list[float]
    type_step: list[Text]


class HistoryBFGS(TypedDict):
    iteration: list[float]
    point: list[Tuple]
    function: list[float]


def update_history_brent(history: HistoryBrent, values: Sequence[Any]) -> HistoryBrent:
    """
    Updates brent history
    :param history: HistoryBrent object in which the update is required
    :param values: Sequence with values: 'iteration', 'f_least', 'f_middle', 'f_largest',  'x_least',
                                         'x_middle', 'x_largest', 'left_bound', 'right_bound', 'type_step'
    :return: updated HistoryBrent
    """
    name: Literal['iteration', 'f_least', 'f_middle', 'f_largest', 'x_least', 'x_middle',
                  'x_largest', 'left_bound', 'right_bound', 'type_step']

    for i, name in enumerate(['iteration', 'f_least', 'f_middle', 'f_largest', 'x_least', 'x_middle',
                              'x_largest', 'left_bound', 'right_bound', 'type_step']):
        history[name].append(values[i])

    return history


class HiddenPrints:
    """
    Object hides print. Working with context manager "with"::

        >>> with HiddenPrints():
        >>>     print("It won't be printed")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class HistoryGD(TypedDict):
    """
    Class with an optimization history of gradient descent methods
    """
    iteration: list
    f_value: list
    f_grad_norm: list
    x: list[Sequence]
    message: Text


def update_history_gd(history: HistoryGD, values: list) -> HistoryGD:
    """
    Update HistoryMDO with values, which contains iteration, f_value, f_grad_norm, x as a list

    :param history: object of HistoryMDO
    :param values: new values that need to append in history in order iteration, f_value, f_grad_norm, x
    :return: updated history
    """
    key: Literal['iteration', 'f_value', 'f_grad_norm', 'x']
    for i, key in enumerate(['iteration', 'f_value', 'f_grad_norm', 'x']):
        history[key].append(values[i])
    return history


def print_verbose(x_k: torch.Tensor,
                  func_k: torch.Tensor | float,
                  verbose: bool,
                  iteration: int,
                  round_precision: int) -> None:
    """
    Prints iteration verbose

    :param x_k: specific point
    :param func_k: float or 0-d tensor
    :param verbose: flag of print verbose
    :param iteration: number of iteration
    :param round_precision: precision of printing float numbers
    :return: none
    """
    round_precision = min(round_precision, 4)
    if verbose:
        print(f'iteration: {iteration:4d}  |  '
              f'x = [{", ".join(map(lambda x: f"{round(float(x), round_precision):>10.4f}", x_k))}]  |  '
              f'f(x) = {round(float(func_k), round_precision)}')
