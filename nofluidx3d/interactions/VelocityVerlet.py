# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 12:41:01 2025

@author: Richard Kellnberger
"""
from pathlib import Path

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

from nofluidx3d.interactions.Interaction import Interaction
from nofluidx3d.KernelBuilder import KernelBuilder
from nofluidx3d.openCL import getContext

kernels = Path(__file__).parents[1] / "kernels"


class VelocityVerlet(Interaction):
    def __init__(self, numPoints, fixTopBottom=False):
        self.fixTopBottom = fixTopBottom

    def createBuffers(self):
        numPoints = self.simulation.numPoints
        # create additional buffers for velocity and old forceB
        velocityNP = np.zeros(3 * numPoints).astype(np.float64)
        forceOldNP = np.zeros(3 * numPoints).astype(np.float64)
        self.velocityB = cl.Buffer(
            getContext(), mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocityNP
        )
        self.forceOldB = cl.Buffer(
            getContext(), mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=forceOldNP
        )

    def build(self):
        self.scope = self.simulation.numPoints
        KernelBuilder.define(INSERT_NUM_POINTS=self.simulation.numPoints)
        if self.fixTopBottom:
            self.knl = KernelBuilder.build(
                kernels / "Interactions" / "VelocityVerlet.cl", "VelocityVerletFixedTopBottom"
            )
        else:
            self.knl = KernelBuilder.build(
                kernels / "Interactions" / "VelocityVerlet.cl", "VelocityVerlet"
            )

    def setArgs(self):
        self.knl.set_args(
            self.simulation.points.buf, self.velocityB, self.simulation.force.buf, self.forceOldB
        )
