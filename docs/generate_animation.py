from sklearn.datasets import make_moons, make_regression

from src.nueramic_mathml.visualize.ml_animation import *
from src.nueramic_mathml.visualize.multi_animation import *
from src.nueramic_mathml.visualize.one_animation import *

color_light = '#F4F4F4'
color_dark = '#212121'


def f(x): return x ** 3 - x ** 2 - x


def f2(x): return x[0] ** 2 + x[1] ** 2 / 2


def make_htmls(type_anim: str = 'gss'):
    if type_anim == 'gss':
        _, h = golden_section_search(f, (0, 2), keep_history=True)
        fig = gen_animation_gss(f, (0, 2), h)

    elif type_anim == 'spi':
        _, h = successive_parabolic_interpolation(f, (0, 2), keep_history=True)
        fig = gen_animation_spi(f, (0, 2), h)

    elif type_anim == 'brent':
        _, h = brent(f, (0, 2), keep_history=True)
        fig = gen_animation_brent(f, h)

    elif type_anim == 'bfgs':
        _, h = bfgs(f2, torch.tensor([8, 5]), keep_history=True)
        fig = gen_animated_surface(f2, h, title='<b>BFGS minimization steps</b>')

    elif type_anim == 'optim':
        _, h = gd_optimal(f2, torch.tensor([8, 5]), keep_history=True)
        fig = gen_simple_gradient(f2, h, title='<b>Gradient descent with optimal step</b>')

    elif type_anim == 'rbf':
        torch.random.manual_seed(7)
        x, y = make_moons(1000, noise=0.15, random_state=7)
        x, y = torch.tensor(x), torch.tensor(y)

        model = LogisticRegressionRBF(x[:50])
        model.fit(x, y, epochs=5000)

        model.metrics_tab(x, y)
        fig = gen_classification_plot(x, y, model, threshold=0.5, epsilon=0.001,
                                      title='<b>Logistic Regression with RBF</b>')

    elif type_anim == 'roc_curve':
        yt = torch.tensor([1, 1, 0, 0, 1, 0])
        yp = torch.tensor([0.7, 0.6, 0.3, 0.5, 0.4, 0.4])
        fig = roc_curve_plot(yt, yp)

    elif type_anim == 'linear':

        x, y = make_regression(200, 1, noise=20, random_state=21)
        x, y = torch.tensor(x), torch.tensor(y)
        regression = LinearRegression().fit(x, y)
        fig = gen_regression_plot(x, y, regression)

    elif type_anim == 'linear4d':
        x, y = make_regression(200, 4, noise=20, random_state=21)
        x, y = torch.tensor(x), torch.tensor(y)
        regression = LinearRegression().fit(x, y)
        fig = gen_regression_plot(x, y, regression)

    else:
        return

    fig = fig.update_layout(
        {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis': {'color': 'black', 'gridcolor': color_light, 'zerolinecolor': color_light},
            'yaxis': {'color': 'black', 'gridcolor': color_light, 'zerolinecolor': color_light},
            'title': {'font': {'color': 'black'}},
            'legend': {'font': {'color': 'black'}},
        }
    )

    fig.write_html(f'./_static/charts/{type_anim.upper()}-animation-light.html')

    fig = fig.update_layout(
        {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'xaxis': {'color': 'white', 'gridcolor': color_dark, 'zerolinecolor': color_dark},
            'yaxis': {'color': 'white', 'gridcolor': color_dark, 'zerolinecolor': color_dark},
            'title': {'font': {'color': 'white'}},
            'legend': {'font': {'color': 'white'}},
        }
    )

    fig.write_html(f'./_static/charts/{type_anim.upper()}-animation-dark.html')


make_htmls('gss')
make_htmls('spi')
make_htmls('brent')
make_htmls('bfgs')
make_htmls('optim')
make_htmls('rbf')
make_htmls('roc_curve')
make_htmls('linear')
make_htmls('linear4d')
