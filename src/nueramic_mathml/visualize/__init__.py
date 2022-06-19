from .one_animation import gen_animation_gss, gen_animation_spi, gen_animation_brent
from .multi_animation import make_surface, make_contour, gen_animated_surface, gen_simple_gradient
from .ml_animation import gen_classification_plot, gen_regression_plot

__all__ = [
    "gen_animation_gss",
    "gen_animation_spi",
    "gen_animation_brent",
    "make_surface",
    "make_contour",
    "gen_animated_surface",
    "gen_simple_gradient",
    "gen_classification_plot",
    "gen_regression_plot"
]
