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
Nueramic-mathml is a library for visualizing and logging the steps of basic optimization algorithms in machine learning. The project uses torch for calculations and plotly for visualization.

.. code-block:: python

  pip install nueramic-mathml


Quick tour  |Colab_1|_
======================

You can minimize the functions and see a detailed description of each step. After minimizing, you have a history with complete logs.

.. code-block:: python

    def f(x): return x ** 3 - x ** 2 - x
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
