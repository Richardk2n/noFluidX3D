# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Tue Mar 25 14:56:31 2025

@author: Richard Kellnberger
"""

import pyopencl as cl

from nofluidx3d.openCL import getCommandQueue


class Interaction:
    def __init__(self, scope):
        self.scope = scope
        self.knl = None

    def beforeTimeStep(self, globalTime):
        pass

    def enqueue(self):
        cl.enqueue_nd_range_kernel(getCommandQueue(), self.knl, (self.scope,), None)
