# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Tue Mar 25 14:56:31 2025

@author: Richard Kellnberger
"""

import pyopencl as cl

from nofluidx3d.openCL import getCommandQueue


class Interaction:

    def __init__(self, scope=None):
        self.scope = scope
        self.knl = None

    def register(self, simulation):
        self.simulation = simulation
        self.createBuffers()
        self.build()
        self.setArgs()

    def createBuffers(self):
        pass

    def build(self):
        pass

    def setArgs(self):
        pass

    def beforeTimeStep(self, globalTime):
        pass

    def enqueue(self):
        cl.enqueue_nd_range_kernel(getCommandQueue(), self.knl, (self.scope,), None)
