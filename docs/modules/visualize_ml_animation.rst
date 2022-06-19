visualize.ml\_animation
===========================
.. currentmodule:: nueramic_mathml.visualize.ml_animation

.. autofunction:: gen_classification_plot

.. raw:: html
    :file: ../_static/charts/RBF-animation-dark.html
    :class: only-dark

.. raw:: html
    :file: ../_static/charts/RBF-animation-light.html
    :class: only-light

----------

.. autofunction:: roc_curve_plot

.. raw:: html
    :file: ../_static/charts/ROC_CURVE-animation-dark.html
    :class: only-dark

.. raw:: html
    :file: ../_static/charts/ROC_CURVE-animation-light.html
    :class: only-light


.. autofunction:: gen_regression_plot

.. raw:: html
    :file: ../_static/charts/LINEAR-animation-dark.html
    :class: only-dark

.. raw:: html
    :file: ../_static/charts/LINEAR-animation-light.html
    :class: only-light

.. code-block:: python3

    >>> # Let's create 4-dimensional data and perform a linear regression.
    >>> # After that, t-sne will show the data on the plane

    >>> x, y = make_regression(200, 4, noise=20, random_state=21)
    >>> x, y = torch.tensor(x), torch.tensor(y)
    >>> regression = LinearRegression().fit(x, y)
    >>> gen_regression_plot(x, y, regression)


.. raw:: html
    :file: ../_static/charts/LINEAR4D-animation-dark.html
    :class: only-dark

.. raw:: html
    :file: ../_static/charts/LINEAR4D-animation-light.html
    :class: only-light

.. code-block:: python3

    >>> regression.metrics_tab(x, y)

.. code-block:: python3

    {'r2': 0.9711183309555054,
     'mae': 15.044872283935547,
     'mse': 365.99530029296875,
     'mape': 55.71377182006836}
