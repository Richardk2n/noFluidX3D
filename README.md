# NoFluidx3d

This python program reuses OpenCL code written for Fluidx3D to calculate AFM Simulations on GPU without a Lattice Boltzmann Fluid.
It still uses Fluid quantities such as viscosity for de-dimensionalization purposes in order to be equivalent to Fluidx3D. It even uses the same configs!



Be aware: this repository does not contain any of the actual opencl code and just hard-references it. This code was absolutely not written to be used by others, and if you have to: May the FSM have mercy on your soul.


Any additional .cl files given are written specifically for noFluidx3d in order to realize things like time-propagation (which are of course not needed for Fluidx3D).
