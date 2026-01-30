# noFluidX3D

This project was used in my doctoral thesis (will be linked here if it is ever published).
This python program reuses OpenCL code written for [FluidX3D](https://github.com/Richardk2n/FluidX3D) to calculate AFM Simulations on GPU without a Lattice Boltzmann fluid.
It still uses fluid quantities such as viscosity for de-dimensionalization purposes in order to be equivalent to Fluidx3D. It even uses the same configs!

Be aware: this repository only contains some of the actual opencl code and hard-references the rest.
This code was absolutely not written to be used by anyone else, and is only known to work with the exakt AFM simulations in my Dissertation.

Any additional .cl files given are written specifically for noFluidx3d in order to realize things like time-propagation (which are of course not needed for FluidX3D).

This project requires Python 3.8 or newer.


## Installing

This is not on PyPi or similar.
To install it, clone the repo, enter the directory and run the following command.

```bash
pip install -e .
```

This installs the local project in a way, that allows edits to the local project.

## Running on 50-1 (Not relevant for the general public, pls ignore)
- Replace the `libOpenCL-b54a1ea0.so.1.0.0` in `~/.myVenv/lib/python3.10/site-packages/pyopencl/.libs/` with `/opt/amdgpu-pro/lib64/libOpenCL.so.1`.
- Start `python` using `LD_PRELOAD=~/gcc14/lib64/libstdc++.so.6` to load a standardlib compatible with `rocm`.


## Developing

Install development dependencies with (you might want to create a virtualenv first):

```bash
pip install -r requirements.txt
```

The project is formatted with [black](https://github.com/psf/black). You can either configure your IDE to automatically format code with it, run it manually (``black .``) or rely on pre-commit (see below) to format files on git commit.

The project is formatted with [isort](https://github.com/PyCQA/isort). You can either configure your IDE to automatically sort imports with it, run it manually (``isort .``) or rely on pre-commit (see below) to sort files on git commit.

This project uses [pre-commit](https://pre-commit.com/) to enforce code-quality. After cloning the repository install the pre-commit hooks with:

```bash
pre-commit install
```

After that pre-commit will run [all defined hooks](.pre-commit-config.yaml) on every ``git commit`` and keep you from committing if there are any errors.

Contirbutions with incorrect formatting are not accepted.
