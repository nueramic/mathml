from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from ..multi_optimize import gd_optimal, bfgs
from ..support import HistoryGD


def make_contour(function: Callable[[torch.Tensor], float | torch.Tensor],
                 bounds: tuple[tuple[float, float], tuple[float, float]],
                 cnt_dots: int = 100,
                 colorscale='teal',
                 showlegend: bool = False) -> go.Contour:
    """
    Return go.Contour for draw by go.Figure. Evaluate function per each point in the 2d grid

    :param function: callable that depends on the first positional argument
    :param bounds: two tuples with constraints for x- and y-axis
    :param cnt_dots: number of point per each axis
    :param colorscale: plotly colorscale for go.Contour
    :param showlegend: show legend flag
    :return: go.Contour

    .. code-block:: python3

        >>> def f(x): return x[0] ** 2 + x[1] ** 2 / 2
        >>> make_contour(f, ((0, 1), (0, 1)), cnt_dots=4)

    .. code-block:: python3

        Contour({
            'colorscale': [[0.0, 'rgb(209, 238, 234)'], [0.16666666666666666, 'rgb(168,
                           219, 217)'], [0.3333333333333333, 'rgb(133, 196, 201)'], [0.5,
                           'rgb(104, 171, 184)'], [0.6666666666666666, 'rgb(79, 144,
                           166)'], [0.8333333333333334, 'rgb(59, 115, 143)'], [1.0,
                           'rgb(42, 86, 116)']],
            'name': 'f(x, y)',
            'showlegend': False,
            'showscale': False,
            'x': array([0.        , 0.33333334, 0.6666666 , 1.        ], dtype=float32),
            'y': array([0.        , 0.33333334, 0.6666666 , 1.        ], dtype=float32),
            'z': array([[0.        , 0.11111112, 0.4444444 , 1.        ],
                        [0.05555556, 0.16666669, 0.49999994, 1.0555556 ],
                        [0.2222222 , 0.3333333 , 0.66666657, 1.2222222 ],
                        [0.5       , 0.6111111 , 0.9444444 , 1.5       ]], dtype=float32)
        })

    """
    x_axis, y_axis, z_axis = _make_axis(function, bounds, cnt_dots)

    return go.Contour(x=x_axis, y=y_axis, z=z_axis, colorscale=colorscale,
                      name='f(x, y)', showscale=showlegend, showlegend=showlegend)


def gen_simple_gradient(function: Callable[[torch.Tensor], torch.Tensor],
                        history: HistoryGD,
                        cnt_dots: int = 200,
                        title: str = '<b>Contour plot with optimization steps</b>',
                        showlegend: bool = True,
                        font_size: int = 18) -> go.Figure:
    """
    Return go.Figure with gradient steps under contour plot. Not animated

    :param function: callable that depends on the first positional argument
    :param history: History after some gradient method
    :param cnt_dots: the numbers of point per each axis
    :param title: title of chart
    :param showlegend: flag of showing legend
    :param font_size: font size
    :return: go.Figure with contour and line of gradient steps

    .. code-block:: python3

        >>> def f(x): return x[0] ** 2 + x[1] ** 2 / 2
        >>> x_opt, hist = gd_optimal(f, torch.tensor([8, 5]), keep_history=True)
        >>> gen_simple_gradient(f, hist).show()

    """

    descent_history = _make_descent_history(history)
    bounds = _make_ranges(history)

    layout = go.Layout(
        title=title,
        xaxis={'title': r'<b>x</b>', 'color': 'black'},
        yaxis={'title': r'<b>y</b>', 'color': 'black'},
        font={'size': font_size, 'color': 'black'},
        legend={'font': {'color': 'black'}}
    )

    contour = make_contour(function=function, bounds=bounds, cnt_dots=cnt_dots, showlegend=False)

    descending_way = go.Scatter(
        x=descent_history.x,
        y=descent_history.y,
        name='descent',
        mode='lines+markers',
        line={'width': 3, 'color': 'rgb(202, 40, 22)'},
        marker={'size': 10, 'color': 'rgb(202, 40, 22)'},
        showlegend=showlegend
    )

    fig = go.Figure(data=[contour, descending_way], layout=layout)

    return fig


def make_surface(function: Callable[[torch.Tensor], float | torch.Tensor],
                 bounds: tuple[tuple[float, float], tuple[float, float]],
                 cnt_dots: int = 100,
                 colorscale='teal',
                 showlegend: bool = False) -> go.Surface:
    """
    Return go.Surface for draw by go.Figure. Evaluate function per each point in the 2d grid

    :param function: callable that depends on the first positional argument
    :param bounds: two tuples with constraints for x- and y-axis
    :param cnt_dots: number of point per each axis
    :param colorscale: plotly colorscale for go.Contour
    :param showlegend: showlegend flag
    :return: go.Surface

    .. code-block:: python3

        >>> def f(x): return x[0] ** 2 + x[1] ** 2 / 2
        >>> make_surface(f, ((0, 1), (0, 1)), cnt_dots=4)

    .. code-block:: python3

        Surface({
            'colorscale': [[0.0, 'rgb(209, 238, 234)'], [0.16666666666666666, 'rgb(168,
                           219, 217)'], [0.3333333333333333, 'rgb(133, 196, 201)'], [0.5,
                           'rgb(104, 171, 184)'], [0.6666666666666666, 'rgb(79, 144,
                           166)'], [0.8333333333333334, 'rgb(59, 115, 143)'], [1.0,
                           'rgb(42, 86, 116)']],
            'name': 'f(x, y)',
            'opacity': 0.75,
            'showlegend': False,
            'x': array([0.        , 0.33333334, 0.6666666 , 1.        ], dtype=float32),
            'y': array([0.        , 0.33333334, 0.6666666 , 1.        ], dtype=float32),
            'z': array([[0.        , 0.11111112, 0.4444444 , 1.        ],
                        [0.05555556, 0.16666669, 0.49999994, 1.0555556 ],
                        [0.2222222 , 0.3333333 , 0.66666657, 1.2222222 ],
                        [0.5       , 0.6111111 , 0.9444444 , 1.5       ]], dtype=float32)
        })

    """

    x_axis, y_axis, z_axis = _make_axis(function, bounds, cnt_dots)

    return go.Surface(x=x_axis, y=y_axis, z=z_axis, colorscale=colorscale,
                      name='f(x, y)', opacity=0.75, showlegend=showlegend)


def _check_bounds(bounds: tuple[tuple[float, float], tuple[float, float]]):
    """
    Assert if bounds not have 2 tuple. In 2 tuples exists 2 floats

    :param bounds: two tuples with constraints for x- and y-axis
    """
    assert len(bounds) == 2, 'two tuples are required'
    assert len(bounds[0]) == 2 and len(bounds[1]) == 2, 'both tuples must have 2 numbers'


def _make_z(function: Callable[[torch.Tensor], float],
            x_axis: torch.Tensor,
            y_axis: torch.Tensor,
            cnt_dots) -> torch.Tensor:
    """
    Returns matrix, with calculated function at each point from grid square bounds[0] x bounds[1]

    :param function: callable that depends on the first positional argument
    :param x_axis: values for x-axis
    :param y_axis: values for y-axis
    :param cnt_dots: number of point per each axis
    :return: tensor with function values
    """

    z_axis = torch.zeros(cnt_dots, cnt_dots).float()
    for i, y in enumerate(y_axis):
        z_axis_i = []
        for x in x_axis:
            z_axis_i.append(function(torch.tensor([x, y])))

        z_axis[i, :] = torch.tensor(z_axis_i).float()

    return z_axis


def _make_axis(function: Callable[[torch.Tensor], float],
               bounds: tuple[tuple[float, float], tuple[float, float]],
               cnt_dots: int) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns x, y, z transformed

    :param function: callable that depends on the first positional argument
    :param bounds: constraints for each axis

    :param cnt_dots: number of point per each axis
    :return: tensor with function values
    :return: three tensor for drawing
    """
    _check_bounds(bounds)
    x_axis = torch.linspace(bounds[0][0], bounds[0][1], cnt_dots)
    y_axis = torch.linspace(bounds[1][0], bounds[1][1], cnt_dots)

    z_axis = _make_z(function, x_axis, y_axis, cnt_dots)

    return x_axis, y_axis, z_axis


def _make_descent_history(history: HistoryGD) -> pd.DataFrame:
    """
    Return converted HistoryGD object into pd.DataFrame with columsn ['x', 'y', 'z', 'iteration']::

        >>> point, hist = gd_optimal(lambda x: x[0] ** 2 + x[1] ** 2 * 1.01, torch.tensor([10, 10]), keep_history=True)
        >>> _make_descent_history(hist).round(4)
        +---+----------+----------+----------+-------------+
        |   | x        | y        | z        | iteration   |
        +===+==========+==========+==========+=============+
        | 0 | 10.0000  | 10.0000  | 201.000  | 0           |
        | 1 | 0.0502   | -0.0493  | 0.005    | 1           |
        | 2 | 0.0002   | 0.0002   | 0.000    | 2           |
        | 3 | 0.0000   | -0.0000  | 0.000    | 3           |
        +---+----------+----------+----------+-------------+

    :param history: History after some gradient method
    :return: pd.DataFrame

    """
    x_points, y_points = torch.stack(history['x']).T

    output_data = pd.DataFrame(
        {
            'x': x_points,
            'y': y_points,
            'z': torch.stack(history['f_value']),
            'iteration': history['iteration']
        }
    )
    return output_data


def _make_ranges(history: HistoryGD, k: float = 0.5) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return bounds for the x-axis and the y-axis.
    1. Find a min_x and max_x and then x_range = [min_x - (max_x - min_x) * k, max_x + (max_x - min_x) * k]
    2. Similarly for y_axis

    :param history: History after some gradient method
    :param k: [min_x - (max_x - min_x) * k, max_x + (max_x - min_x) * k]
    :return: [x_range, y_range]
    """

    descent_history = _make_descent_history(history)
    min_x = descent_history.x.min() - 0.1
    max_x = descent_history.x.max() + 0.1

    min_y = descent_history.y.min() - 0.1
    max_y = descent_history.y.max() + 0.1

    x_range = (min_x - (max_x - min_x) * k, max_x + (max_x - min_x) * k)
    y_range = (min_y - (max_y - min_y) * k, max_y + (max_y - min_y) * k)

    return x_range, y_range


def _make_descent_frames_3d(function: Callable[[torch.Tensor], float],
                            history: HistoryGD,
                            number_points: int = 10) -> list[go.Frame]:
    """
    Make sequence of go.Frame which contain frame for each step of descent with a previous history

    :param function: callable that depends on the first positional argument
    :param history: History after some gradient method
    :param number_points: the number of points in will be calculated by a function from (x_i, y_i) to  (x_i+1, y_i+1)
    :return: list[go.Frame]
    """

    frames = []
    descent_history = _make_descent_history(history)

    draw_descent = [[], [], []]

    # For each step of optimization (minimization steps)
    for i in range(descent_history.shape[0]):

        if i > 0:
            # it will be calculated way from point(x_i, y_i) to point (x_i+1, y_i+1)
            x0, x1 = descent_history.x[i - 1], descent_history.x[i]
            y0, y1 = descent_history.y[i - 1], descent_history.y[i]

            if (x0 - x1) ** 2 + (y0 - y1) ** 2 > 0.5:
                num_p = number_points
            else:
                num_p = 2

            for alpha in torch.linspace(0, 1, num_p):
                draw_descent[0].append(x0 * alpha + x1 * (1 - alpha))
                draw_descent[1].append(y0 * alpha + y1 * (1 - alpha))
                draw_descent[2].append(function(torch.tensor([draw_descent[0][-1], draw_descent[1][-1]])))
            else:
                draw_descent[0].append(torch.nan)
                draw_descent[1].append(torch.nan)
                draw_descent[2].append(torch.nan)

        scatter_line = go.Scatter3d(x=draw_descent[0],
                                    y=draw_descent[1],
                                    z=draw_descent[2],
                                    name='descent',
                                    mode='lines',
                                    line={'width': 4, 'color': 'rgb(1, 23, 47)'})

        scatter_points = go.Scatter3d(x=descent_history.x[:i + 1],
                                      y=descent_history.y[:i + 1],
                                      z=descent_history.z[:i + 1],
                                      name='descent',
                                      mode='markers',
                                      marker={'size': 4, 'color': 'rgb(1, 23, 47)'},
                                      showlegend=False)

        frames.append(go.Frame(data=[scatter_points, scatter_line], name=i, traces=[1, 2]))

    return frames


def gen_animated_surface(function: Callable[[torch.Tensor], float],
                         history: HistoryGD,
                         cnt_dots: int = 100,
                         title: str = '<b>Surface with optimization steps</b>') -> go.Figure:
    """
    Return go.Figure with animation per each step of descent

    :param function: callable that depends on the first positional argument
    :param history: History after some gradient method
    :param cnt_dots: the numbers of point per each axis
    :param title: how many frames will drawing. ~300 frames will be drawn for ~5-10 seconds
    :return: go.Figure with animation steps on surface

    .. code-block:: python3

        >>> def f(x): return x[0] ** 2 + x[1] ** 2 / 9
        >>> _, h = bfgs(f, torch.tensor([10, 10]), keep_history=True)

    """
    descent_history = _make_descent_history(history)
    bounds = _make_ranges(history)

    first_point = go.Scatter3d(x=descent_history.x[:1],
                               y=descent_history.y[:1],
                               z=descent_history.z[:1],
                               mode='markers',
                               marker={'size': 4, 'color': 'rgb(1, 23, 47)'},
                               showlegend=False)

    surface = make_surface(function, bounds, cnt_dots=cnt_dots)
    layout = px.scatter_3d(descent_history, x='x', y='y', z='z', animation_frame='iteration').layout
    frames = _make_descent_frames_3d(function, history)

    fig = go.Figure(data=[surface, first_point, first_point],
                    layout=layout,
                    frames=frames)
    fig.update_scenes(
        xaxis_title=r'<b>x</b>',
        yaxis_title=r'<b>y</b>',
        zaxis_title=r'<b>z</b>',
    )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.25, y=-1.25, z=1.25)
    )

    fig.update_layout(
        {
            'title': {
                'font': {'size': 22, 'color': 'black'},
                'text': title
            },
            'scene_camera': camera
        }
    )
    return fig
