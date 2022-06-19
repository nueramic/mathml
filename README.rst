.. -*- mode: rst -*-
   
|PyPi|_ |Python|_ |Download|_ |License|_ |RTD|_


------

.. |PyPi| image:: https://img.shields.io/pypi/v/nueramic-mathml?color=edf2f4&style=flat-square
.. _PyPi: https://pypi.org/project/nueramic-mathml/


.. |Python| image:: https://img.shields.io/pypi/pyversions/p?color=edf2f4&style=flat-square
.. _Python: https://github.com/nueramic/mathml

.. |Download| image:: https://img.shields.io/pypi/dm/nueramic-mathml?color=edf2f4&label=dowloads&style=flat-square
.. _Download: https://pypi.org/project/nueramic-mathml/

.. |License| image:: https://img.shields.io/github/license/nueramic/mathml?color=edf2f4&style=flat-square
.. _License: https://github.com/nueramic/mathml

.. |RTD| image:: https://img.shields.io/readthedocs/nueramic-mathml?color=edf2f4&style=flat-square
.. _RTD: https://nueramic-mathml.readthedocs.io

.. |Colab_1| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _Colab_1: https://colab.research.google.com/drive/19moQvDMK8kfTDYOGuRwEl06jdf_KXNMW?usp=sharing


.. raw:: html

   <p align="center">
   <picture align="center">
     <source width=150px" media="(prefers-color-scheme: dark)" srcset="docs/_static/nueramic-logo-cropped-white.svg">
     <source width=150px" media="(prefers-color-scheme: light)" srcset="docs/_static/nueramic-logo-cropped-black.svg">
     <img alt="two logos" src="docs/_static/nueramic-logo-cropped-black.svg">
   </picture>
   </p>


Nueramic MathML
===============
Nueramic-mathml is a library for visualizing and logging the steps of optimization algorithms in machine learning. The project uses torch for calculations and plotly for visualization.

.. code-block:: python

    pip install nueramic-mathml


Quick tour  |Colab_1|_
======================

Optimization
------------------
You can minimize the functions and see a detailed description of each step. After minimizing, you have a history with complete logs.
Also available multidimensional optimisation.

.. code-block:: python

    def f(x): return x ** 3 - x ** 2 - x  # Minimum at x = 1
    bounds = (0, 3)
    one_optimize.golden_section_search(f, bounds, epsilon=0.01, verbose=True)[0]

    Iteration: 0 	|	 point = 1.500 	|	 f(point) = -0.375
    Iteration: 1 	|	 point = 0.927 	|	 f(point) = -0.990
    Iteration: 2 	|	 point = 1.281 	|	 f(point) = -0.820
    Iteration: 3 	|	 point = 1.062 	|	 f(point) = -0.992
    Iteration: 4 	|	 point = 0.927 	|	 f(point) = -0.990
    Iteration: 5 	|	 point = 1.011 	|	 f(point) = -1.000
    Iteration: 6 	|	 point = 0.959 	|	 f(point) = -0.997
    Iteration: 7 	|	 point = 0.991 	|	 f(point) = -1.000
    Iteration: 8 	|	 point = 1.011 	|	 f(point) = -1.000
    Iteration: 9 	|	 point = 0.998 	|	 f(point) = -1.000
    Iteration: 10 	|	 point = 1.006 	|	 f(point) = -1.000
    Searching finished. Successfully. code 0
    1.0059846881033916

Models
-------
You can use our models for classification and regression

.. code-block:: python

    from nueramic_mathml.ml import LogisticRegressionRBF
    from sklearn.datasets import make_moons

    x, y = make_moons(10_000, noise=.1, random_state=84)
    x, y = torch.tensor(x), torch.tensor(y)
    logistic_model_rbf = LogisticRegressionRBF(x[:1000]).fit(x, y, show_epoch=10)

    Epoch:     1 | CrossEntropyLoss:  0.71496
    Epoch:    12 | CrossEntropyLoss:  0.35328
    Epoch:    23 | CrossEntropyLoss:  0.27769
    Epoch:    34 | CrossEntropyLoss:  0.22395
    Epoch:    45 | CrossEntropyLoss:  0.19266
    Epoch:    56 | CrossEntropyLoss:  0.16695
    Epoch:    67 | CrossEntropyLoss:  0.14686
    Epoch:    78 | CrossEntropyLoss:  0.13051
    Epoch:    89 | CrossEntropyLoss:  0.11724
    Epoch:   100 | CrossEntropyLoss:  0.10629

    logistic_model_rbf.metrics_tab(x, y)

    {'auc_roc': 0.9974513817072977,
     'f1': 0.9700730618209839,
     'precision': 0.9709476828575134,
     'recall': 0.9692000150680542}

Visualizations
---------------
You can create beautiful animations of optimization algorithms and
regression/classification models.

.. raw:: html

       <p align="center">
       <picture align="center">
         <source width=150px" media="(prefers-color-scheme: dark)" srcset="docs/_static/charts/RBF-animation-light.html">
         <source width=150px" media="(prefers-color-scheme: light)" srcset="docs/_static/charts/RBF-animation-dark.html">
         <img alt="rbf" src="docs/_static/nueramic-logo-cropped-black.svg">
       </picture>
       </p>
