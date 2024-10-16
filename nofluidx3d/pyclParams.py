import pyopencl as cl

from nofluidx3d.openCL import getCommandQueue, getContext, getDevice

ctx = getContext()
queue = getCommandQueue()
mf = cl.mem_flags
