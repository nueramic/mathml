from __future__ import annotations

from typing import List, TypedDict, Text, Tuple, Sequence, Any, Literal


class HistoryGSS(TypedDict):
    """
    Class with an optimization history of Golden section search
    """
    iteration: List
    middle_point: List[float]
    f_value: List[float]
    left_point: List[float]
    right_point: List[float]


class HistorySPI(TypedDict):
    """
    Class with an optimization history of Successive parabolic interpolation
    """
    iteration: List[int]
    f_value: List[float]
    x0: List[float]
    x1: List[float]
    x2: List[float]


class HistoryBrent(TypedDict):
    """
    Class with an optimization history of Brant's algorithm
    """
    iteration: List[int]

    f_least: List[float]
    f_middle: List[float]
    f_largest: List[float]

    x_least: List[float]
    x_middle: List[float]
    x_largest: List[float]

    left_bound: List[float]
    right_bound: List[float]
    type_step: List[Text]


class HistoryBFGS(TypedDict):
    iteration: List[float]
    point: List[Tuple]
    function: List[float]


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
