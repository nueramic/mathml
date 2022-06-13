multi\_optimize
================

.. toctree::
   :glob:
   :maxdepth: 4
   :caption: Methods

   multi-optimize/gd-constant.rst
   multi-optimize/gd-fractional.rst
   multi-optimize/gd-optimal.rst
   multi-optimize/gd-nonlinear_conjugate.rst
   multi-optimize/bfgs.rst
   multi-optimize/constr-lagrangian.rst
   multi-optimize/log-barrier.rst
   multi-optimize/primal-dual.rst

Methods to solve the problem:


.. math::

    f(x_0, x_1, \dots x_n) \rightarrow \min

.. graphviz::
    :name: sphinx.ext.graphviz
    :caption: Sphinx and GraphViz Data Flow
    :alt: How Sphinx and GraphViz Render the Final Document
    :align: center

     digraph "sphinx-ext-graphviz" {
         size="6,4";
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12"];
         edge [fontname="Sans", fontsize="9"];

         sphinx [label="Sphinx", shape="component",
                   href="https://www.sphinx-doc.org/",
                   target="_blank"];
         dot [label="GraphViz", shape="component",
              href="https://www.graphviz.org/",
              target="_blank"];
         docs [label="Docs (.rst)", shape="folder",
               fillcolor=green, style=filled];
         svg_file [label="SVG Image", shape="note", fontcolor=white,
                 fillcolor="#3333ff", style=filled];
         html_files [label="HTML Files", shape="folder",
              fillcolor=yellow, style=filled];

         docs -> sphinx [label=" parse "];
         sphinx -> dot [label=" call ", style=dashed, arrowhead=none];
         dot -> svg_file [label=" draw "];
         sphinx -> html_files [label=" render "];
         svg_file -> html_files [style=dashed];
     }
