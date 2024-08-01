noFluidX3D
======================
#.. image:: https://badge.fury.io/py/pylsp-mypy.svg
#    :target: https://badge.fury.io/py/pylsp-mypy

#.. image:: https://github.com/python-lsp/pylsp-mypy/workflows/Python%20package/badge.svg?branch=master
#    :target: https://github.com/python-lsp/pylsp-mypy/

This python program reuses OpenCL code written for Fluidx3D to calculate AFM Simulations on GPU without a Lattice Boltzmann Fluid.
It still uses Fluid quantities such as viscosity for de-dimensionalization purposes in order to be equivalent to Fluidx3D. It even uses the same configs!



Be aware: this repository does not contain any of the actual opencl code and just hard-references it. This code was absolutely not written to be used by others, and if you have to: May the FSM have mercy on your soul.


Any additional .cl files given are written specifically for noFluidx3d in order to realize things like time-propagation (which are of course not needed for Fluidx3D).


It requires Python 3.8 or newer.


Developing
-------------

Install development dependencies with (you might want to create a virtualenv first):

::

   pip install -r requirements.txt

The project is formatted with `black`_. You can either configure your IDE to automatically format code with it, run it manually (``black .``) or rely on pre-commit (see below) to format files on git commit.

The project is formatted with `isort`_. You can either configure your IDE to automatically sort imports with it, run it manually (``isort .``) or rely on pre-commit (see below) to sort files on git commit.

The project uses two rst tests in order to assure uploadability to pypi: `rst-linter`_ as a pre-commit hook and `rstcheck`_ in a GitHub workflow. This does not catch all errors.

This project uses `pre-commit`_ to enforce code-quality. After cloning the repository install the pre-commit hooks with:

::

   pre-commit install

After that pre-commit will run `all defined hooks`_ on every ``git commit`` and keep you from committing if there are any errors.

.. _black: https://github.com/psf/black
.. _isort: https://github.com/PyCQA/isort
.. _rst-linter: https://github.com/Lucas-C/pre-commit-hooks-markup
.. _rstcheck: https://github.com/myint/rstcheck
.. _pre-commit: https://pre-commit.com/
.. _all defined hooks: .pre-commit-config.yaml
