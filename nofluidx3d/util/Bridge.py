# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 14:26:21 2025

@author: Richard Kellnberger
"""

import pyopencl as cl
from pyopencl import mem_flags as mf

from nofluidx3d.openCL import getCommandQueue, getContext


class Bridge:
    def __init__(self, startarray, datatype):
        self.arrayC = startarray.astype(datatype)
        self.buf = cl.Buffer(getContext(), mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.arrayC)

    def readFromGPU(self):
        cl.enqueue_copy(getCommandQueue(), self.arrayC, self.buf)
        return self.arrayC

    def writeToGPU(self):
        cl.enqueue_copy(getCommandQueue(), self.buf, self.arrayC)
