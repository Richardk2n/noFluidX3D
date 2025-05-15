# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu May 15 13:03:48 2025

@author: Richard Kellnberger
"""
import ctypes
from pathlib import Path

from nofluidx3d.interactions.Interaction import Interaction
from nofluidx3d.KernelBuilder import KernelBuilder

kernels = Path(__file__).parents[1] / "kernels"


class Sphere(Interaction):
    def __init__(self, radius, sphereFunc, forceConst):
        self.radius = radius
        self.sphereFunc = sphereFunc
        self.forceConst = forceConst

        self.globalTime = 0

    def build(self):
        self.scope = self.simulation.numPoints
        KernelBuilder.define(
            INSERT_NUM_POINTS=self.simulation.numPoints, def_FORCE_CONST=self.forceConst
        )
        self.knl = KernelBuilder.build(kernels / "Interactions" / "Sphere.cl", "Interaction_Sphere")

    def setArgs(self):
        self.knl.set_args(
            self.simulation.force.buf,
            self.simulation.points.buf,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.sphereFunc(self.globalTime)),
        )

    def beforeTimeStep(self, globalTime):
        self.globalTime = globalTime
        self.setArgs()
