from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from ..one_optimize import golden_section_search, successive_parabolic_interpolation, brent
from ..support import HistoryGSS, HistorySPI, HistoryBrent

# Color palette
# https://coolors.co/94affd-ff616b-0a50ff-7f8082-50c878

COLOR1 = FRENCH_SKY_BLUE = '#94AFFD'
COLOR2 = FIERY_ROSE = '#FF616B'
COLOR3 = BLUE_RYB = '#0a50ff'
COLOR4 = GRAY_WEB = '#7F8082'
COLOR5 = EMERALD = '#50C878'

standard_layout = dict(
    xaxis_title=r'<b>x</b>',
    yaxis_title=r'<b>f(x)</b>',
    font=dict(size=14),
    legend_title_text='<b>variable names</b>'
)


def transfer_history_gss(history: HistoryGSS, func) -> pd.DataFrame:
    """
    Generate data for plotly express with using animation_frame for animate

    .. code-block:: python3

        >>> def f(x): return x ** 2
        >>> _, hist = golden_section_search(f, (-1, 2), keep_history=True)
        >>> data_for_plot = transfer_history_gss(hist, f)
        Searching finished. Successfully. code 0

    .. code-block:: python3

        >>> data_for_plot[::30]

    +-----+------------+---------+-----------+-----------+-------+
    |     | iteration  | type    | x         | y         | size  |
    +=====+============+=========+===========+===========+=======+
    | 0   | 0          | middle  | 0.500000  | 0.250000  | 3     |
    +-----+------------+---------+-----------+-----------+-------+
    | 30  | 4          | left    | -0.291796 | 0.085145  | 3     |
    +-----+------------+---------+-----------+-----------+-------+
    | 60  | 8          | right   | 0.042572  | 0.001812  | 3     |
    +-----+------------+---------+-----------+-----------+-------+

    :param history: a history object. a dict with lists. keys iteration, f_value, middle_point, left_point, right_point
    :param func: the functions for which the story was created

    :return: pd.DataFrame for gen_animation_gss. index - num of iteration.
    """

    n = len(history['middle_point'])

    df_middle = pd.DataFrame({'iteration': history['iteration'],
                              'type': ['middle'] * n,
                              'x': history['middle_point'],
                              'y': history['f_value'],
                              'size': [3] * n})  # Data for frames for middle point

    df_left = pd.DataFrame({'iteration': history['iteration'],
                            'type': ['left'] * n,
                            'x': history['left_point'],
                            'y': list(map(func, history['left_point'])),
                            'size': [3] * n})  # Data for frames for left point

    df_right = pd.DataFrame({'iteration': history['iteration'],
                             'type': ['right'] * n,
                             'x': history['right_point'],
                             'y': list(map(func, history['right_point'])),
                             'size': [3] * n})  # Data for frames for right point

    df = pd.concat([df_middle, df_left, df_right]).reset_index(drop=True)  # Concatenate all

    return df


def gen_animation_gss(func: Callable,
                      bounds: tuple[float, float],
                      history: HistoryGSS,
                      **kwargs) -> go.Figure:
    """
    Generates an animation of the golden-section search on `func` between the `bounds`

    :param func: callable that depends on the first positional argument
    :param bounds: tuple with left and right points on the x-axis
    :param history: a history object. a dict with lists. keys iteration, f_value, middle_point, left_point, right_point
    :return: go.Figure with graph

    .. code-block:: python3

        >>> def f(x): return x ** 3 - x ** 2 - x
        >>> _, h = golden_section_search(f, (0, 2), keep_history=True)
        >>> gen_animation_gss(f, (0, 2), h)

    """
    x_axis = np.linspace(bounds[0], bounds[1], 500)
    f_axis = np.zeros_like(x_axis)
    diff_x = max(x_axis) - min(x_axis)

    for i, _x in enumerate(x_axis):
        f_axis[i] = func(_x, **kwargs)
    diff_f = max(f_axis) - min(f_axis)

    df = transfer_history_gss(history, func)
    fig = px.scatter(df, x='x', y='y', size='size', color='type',
                     animation_frame='iteration',
                     range_x=[min(x_axis) - diff_x * 0.15, max(x_axis) + diff_x * 0.15],
                     range_y=[min(f_axis) - diff_f * 0.15, max(f_axis) + diff_f * 0.15],
                     size_max=10,
                     title='<b>Golden section search</b>')

    fig.add_trace(go.Scatter(x=x_axis, y=f_axis, name='function'))

    fig.update_layout(**standard_layout)

    return fig


def gen_animation_spi(func: Callable[[float], float],
                      bounds: [float, float],
                      history: HistorySPI) -> go.Figure:
    """
    Generate animation. Per each iteration we create a go.Frame with parabola plot passing through three points

    :param history: a history object. a dict with lists. keys iteration, f_value, middle_point, left_point, right_point
    :param bounds: tuple with left and right points on the x-axis
    :param func: the functions for which the story was created

    .. code-block:: python3

        >>> def f(x): return x ** 3 - x ** 2 - x
        >>> _, h = successive_parabolic_interpolation(f, (0, 2), keep_history=True)
        >>> gen_animation_spi(f, (0, 2), h)

    """

    n = len(history['iteration'])
    x_axis = torch.linspace(bounds[0], bounds[1], 200)
    f_axis = torch.zeros_like(x_axis)
    diff_x = max(x_axis) - min(x_axis)

    for i, _x in enumerate(x_axis):
        f_axis[i] = func(_x)

    diff_f = max(f_axis) - min(f_axis)
    x_range = [x_axis[0] - diff_x * 0.1, x_axis[-1] + diff_x * 0.1]
    f_range = [min(f_axis) - diff_f * 0.25, max(f_axis) + diff_f * 0.25]

    history = pd.DataFrame(history)
    a, b, c = parabolic_coefficients(
        history.loc[0, 'x0'],
        history.loc[0, 'x1'],
        history.loc[0, 'x2'],
        func
    )

    x0, x1, x2 = history.loc[0, ['x0', 'x1', 'x2']].values

    data = [
        go.Scatter(
            x=x_axis, y=float(a) * x_axis ** 2 + float(b) * x_axis + float(c),
            name='parabola 1', marker={'color': 'rgba(55, 101, 164, 1)'}
        ),
        go.Scatter(
            x=[float(x0)], y=[func(float(x0))],
            name='x0', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}
        ),
        go.Scatter(
            x=[float(x1)], y=[func(float(x1))],
            name='x1', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}
        ),
        go.Scatter(
            x=[float(x2)], y=[func(float(x2))],
            name='x2', mode='markers', marker={'size': 10, 'color': 'rgba(231, 29, 54, 1)'}
        ),
        go.Scatter(
            x=x_axis, y=float(a) * x_axis ** 2 + float(b) * x_axis + float(c),
            name='parabola 0', marker={'color': 'rgba(55, 101, 164, 0.3)'},
        ),
        go.Scatter(
            x=x_axis, y=f_axis, name='function', marker={'color': 'rgba(225, 81, 85, 0.8)'}
        )
    ]

    layout = _make_layout(n, x_range, f_range, 'Successive parabolic search')

    frames = []

    x0, x1, x2 = history.loc[0, ['x0', 'x1', 'x2']].values
    a, b, c = parabolic_coefficients(x0, x1, x2, func)
    parabola_pre = float(a) * x_axis ** 2 + float(b) * x_axis + float(c)
    for i in range(0, history.shape[0]):
        x0, x1, x2 = history.loc[i, ['x0', 'x1', 'x2']].values
        a, b, c = parabolic_coefficients(x0, x1, x2, func)

        parabola_new = float(a) * x_axis ** 2 + float(b) * x_axis + float(c)
        frames.append(go.Frame({
            'data': [
                go.Scatter(
                    x=x_axis, y=parabola_new, name=f'parabola {i + 1}',
                    marker={'color': 'rgba(55, 101, 164, 1)'}
                ),
                go.Scatter(
                    x=[float(x0)], y=[func(float(x0))],
                    name='x0', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}
                ),
                go.Scatter(
                    x=[float(x1)], y=[func(float(x1))],
                    name='x1', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}
                ),
                go.Scatter(
                    x=[float(x2)], y=[func(float(x2))],
                    name='x2', mode='markers', marker={'size': 10, 'color': 'rgba(231, 29, 54, 1)'}
                ),
                go.Scatter(
                    x=x_axis, y=parabola_pre, name=f'parabola {max(i, 1)}',
                    marker={'color': 'rgba(55, 101, 164, 0.3)'}, mode='lines'
                )
            ],

            'name': f'{i}'}))
        parabola_pre = parabola_new

    fig = go.Figure(data=data, layout=layout, frames=frames)
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=f_range)
    fig.update_layout(**standard_layout)

    return fig


def parabolic_coefficients(x0: float, x1: float, x2: float, func: Callable[[float], float]) -> [float, float, float]:
    """
    Returns a parabolic function passing through the specified points x0, x1, x2 coeficients

    .. code-block:: python3

        >>> parabolic_coefficients(0, 1, 2, lambda x: x ** 2)
        (1.0, 0.0, 0.0)

    .. math::

        \\begin{bmatrix} a \\\\ b \\\\ c \\end{bmatrix} = \\begin{bmatrix} x_0^2 & x_0 & 1 \\\\ x_1^2 & x_1 & 1 \\\\
        x_2^2 & x_2 & 1 \\end{bmatrix}^{-1} \\cdot \\begin{bmatrix} y_0 \\\\ y_1 \\\\ y_2 \\end{bmatrix}

    :param x0: first point
    :param x1: second point
    :param x2: third point
    :param func: the functions for which the story was created
    :return: coefficients of the parabolic function
    """
    assert x0 != x1 and x0 != x2 and x1 != x2, 'x0, x1, x2 must be different'
    x_mat = torch.tensor([[x0 ** 2, x0, 1.], [x1 ** 2, x1, 1.], [x2 ** 2, x2, 1.]]).double()
    y_mat = torch.tensor([[func(x0)], [func(x1)], [func(x2)]]).double()
    a, b, c = map(float, (torch.linalg.inv(x_mat) @ y_mat).flatten())

    return a, b, c


def _init_function_plot(bounds: list[float, float],
                        func: Callable[[float], float]) -> go.Scatter:
    """
    Returns initial function plot

    :param bounds: left and right constants
    :param func: initial optimization function
    """
    bounds = list(bounds)
    bounds.sort()
    delta = bounds[1] - bounds[0]

    x = torch.linspace(bounds[0] - delta * 0.1, bounds[1] + delta * 0.1, 100)
    fx = list(map(func, x))

    return go.Scatter(x=x, mode='lines', y=fx, name='function', marker={'size': 5, 'color': COLOR1})


def _make_golden_frame(x: list[float, float, float],
                       func: Callable[[float], float],
                       func_plot: go.Scatter,
                       i: int) -> go.Frame:
    """
    Returns golden step frame. Also, first step (initialization)

    :param x: list of x. x[0] - left bound, x[1] - middle, x[2] - right bound
    :param func: initial function
    :param func_plot: go.Scatter of initial function
    :param i: frame's number
    :return: one frame of golden section search step
    """
    x = sorted(list(x), key=func, reverse=True)
    fx = list(map(func, x))

    data = [
        go.Scatter(x=x[:1], y=fx[:1], name='x2', marker={'size': 10, 'color': COLOR3}, mode='markers'),
        go.Scatter(x=x[1:2], y=fx[1:2], name='x1', marker={'size': 10, 'color': COLOR3}, mode='markers'),
        go.Scatter(x=x[2:3], y=fx[2:3], name='x0', marker={'size': 10, 'color': COLOR5}, mode='markers'),
        func_plot,
        go.Scatter(x=[0], y=[0], name='golden step', mode='lines', marker={'size': 10, 'color': 'rgba(0, 0, 0, 0)'})
    ]
    return go.Frame(data=data, name=f'{i}')


def _make_parabolic_frame(x: list[float, float, float],
                          func: Callable[[float], float],
                          func_plot: go.Scatter,
                          i: int) -> go.Frame:
    """
    Returns successive step frame.

    :param x: list of x. x[0] - left bound, x[1] - middle, x[2] - right bound
    :param func: initial function
    :param func_plot: go.Scatter of initial function
    :param i: frame's number
    :return: one frame of successive parabolic step
    """
    a, b, c = parabolic_coefficients(*x, func=func)
    x = sorted(list(x), key=func, reverse=True)
    fx = list(map(func, x))
    x_axis = func_plot.x
    y_axis = a * x_axis ** 2 + b * x_axis + c

    data = [
        go.Scatter(x=x[:1], y=fx[:1], name='x2', marker={'size': 10, 'color': COLOR3}, mode='markers'),
        go.Scatter(x=x[1:2], y=fx[1:2], name='x1', marker={'size': 10, 'color': COLOR3}, mode='markers'),
        go.Scatter(x=x[2:3], y=fx[2:3], name='x0', marker={'size': 10, 'color': COLOR5}, mode='markers'),
        func_plot,
        go.Scatter(
            x=x_axis, y=y_axis, name='parabolic step',
            marker={'size': 8, 'color': COLOR2}, mode='lines', showlegend=True
        ),
    ]
    return go.Frame(data=data, name=f'{i}')


def _make_frames(history: HistoryBrent,
                 func: Callable[[float], float]) -> list:
    """
    Returns frames of brent optimization

    :param history: History of optimization by brent algorithm
    :param func: initial function
    :return: frames of brent optimization
    """
    hdf = pd.DataFrame(history)

    initial_function = _init_function_plot(hdf.loc[0, ['left_bound', 'right_bound']], func)
    frames = []

    for i in range(1, hdf.shape[0]):
        if hdf.loc[i, 'type_step'] == 'golden':
            frame = _make_golden_frame(
                hdf.loc[i, ['left_bound', 'x_least', 'right_bound']],
                func,
                initial_function,
                i
            )
        else:
            frame = _make_parabolic_frame(
                hdf.loc[i, ['x_least', 'x_middle', 'x_largest']],
                func,
                initial_function,
                i
            )

        frames.append(frame)

    return frames


def _make_layout(n: int, x_range: list[float, float], f_range: list[float, float], title: str) -> go.Layout:
    """
    Returns go.Layout for spi and brent algorithms

    :param n: number of frames
    :param x_range: constraints of x
    :param f_range: constraints of y (second axi)
    :param title: title of layout
    :return: go layout with sliders
    """
    layout = go.Layout(
        {
            'font': {'size': 14},
            'legend': {'title': {'text': '<b>variable names</b>'}},
            'sliders': [
                {
                    'active': 0,
                    'currentvalue': {'prefix': 'iteration='},
                    'len': 0.9,
                    'pad': {'b': 10, 't': 60},
                    'steps': [
                        {
                            'args': [
                                [f'{i}'],
                                {
                                    'frame': {'duration': 500, 'redraw': False, 'mode': 'immediate'},
                                    'mode': 'immediate',
                                    'fromcurrent': False,
                                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                                }
                            ],
                            'label': f'{i}',
                            'method': 'animate'
                        } for i in range(n)
                    ],
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top'
                }
            ],

            'title': {'text': f'<b>{title}</b>'},
            'updatemenus': [
                {'buttons': [
                    {
                        'args': [
                            None,
                            {
                                'frame': {'duration': 1000, 'redraw': True},
                                'mode': 'immediate',
                                'fromcurrent': True,
                                'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                            }
                        ],
                        'label': '&#9654;',
                        'method': 'animate'
                    },
                    {
                        'args': [
                            [None],
                            {
                                'frame': {'duration': 100, 'redraw': True},
                                'mode': 'immediate',
                                'fromcurrent': True,
                                'transition': {'duration': 100, 'easing': 'cubic-in-out'}
                            }
                        ],
                        'label': '&#9724;',
                        'method': 'animate'
                    }
                ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 70},
                    'showactive': True,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }
            ],

            'xaxis': {
                'anchor': 'y',
                'domain': [0.0, 1.0],
                'range': x_range,
                'autorange': False,
                'title': {'text': '<b>x</b>'}
            },
            'yaxis': {
                'anchor': 'x',
                'domain': [0.0, 1.0],
                'range': f_range,
                'autorange': False,
                'title': {'text': '<b>f(x)</b>'}
            }
        }
    )
    return layout


def gen_animation_brent(func: Callable[[float], float], history: HistoryBrent) -> go.Figure:
    """
    Returns a visualization of the Brent algorithm. Each iteration shows which iteration.

    :param func: callable that depends on the first positional argument
    :param history: brent optimization history
    :return: animation of optimization

    .. code-block:: python3

        >>> def f(x): return x ** 3 - x ** 2 - x
        >>> _, h = brent(f, (0, 2), keep_history=True)
        >>> gen_animation_brent(f, h)

    """

    # x bounds
    lb, rb, xl = history['left_bound'][0], history['right_bound'][0], history['x_least'][0]
    delta = abs(rb - lb)
    x_range = [lb - delta * 0.2, rb + delta * 0.2]

    # initial frame and data
    hdf = pd.DataFrame(history)
    initial_function = _init_function_plot([lb, rb], func)
    data = _make_golden_frame([lb, xl, rb], func, initial_function, 0).data

    # all frames
    frames = _make_frames(history, func)

    # y bounds
    y_bottom, y_top = min(initial_function.y), max(initial_function.y)
    delta = y_top - y_bottom
    y_range = [y_bottom - delta * 0.25, y_top + delta * 0.25]

    # layout
    layout = _make_layout(hdf.shape[0], x_range, y_range, 'Brent optimization')

    # all visualization
    return go.Figure(data=data, layout=layout, frames=frames)
