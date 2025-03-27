# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 12:20:26 2025

@author: Richard Kellnberger
"""
import ctypes
from pathlib import Path

from nofluidx3d.interactions.Interaction import Interaction
from nofluidx3d.KernelBuilder import KernelBuilder

kernels = Path(__file__).parents[1] / "kernels"


class PlaneAFM(Interaction):
    def __init__(self, topWallFunc, bottomWallFunc, forceConst):
        self.topWallFunc = topWallFunc
        self.bottomWallFunc = bottomWallFunc
        self.forceConst = forceConst

        self.globalTime = 0

    def build(self):
        self.scope = self.simulation.numPoints
        KernelBuilder.define(INSERT_NUM_POINTS=self.simulation.numPoints)
        self.knl = KernelBuilder.build(
            kernels / "Interactions" / "PlaneAFM.cl", "Interaction_PlaneAFM"
        )

    def setArgs(self):
        self.knl.set_args(
            self.simulation.force.buf,
            self.simulation.points.buf,
            ctypes.c_double(self.topWallFunc(self.globalTime)),
            ctypes.c_double(self.bottomWallFunc(self.globalTime)),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.globalTime = globalTime
        self.setArgs()
