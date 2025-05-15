# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu May 15 14:21:21 2025

@author: Richard Kellnberger
"""
import ctypes
from pathlib import Path

from nofluidx3d.interactions.Interaction import Interaction
from nofluidx3d.KernelBuilder import KernelBuilder

kernels = Path(__file__).parents[1] / "kernels"


class Substrate(Interaction):
    def __init__(self, wallPosition, forceConst):
        self.wallPosition = wallPosition
        self.forceConst = forceConst

    def build(self):
        self.scope = self.simulation.numPoints
        KernelBuilder.define(INSERT_NUM_POINTS=self.simulation.numPoints)
        self.knl = KernelBuilder.build(
            kernels / "Interactions" / "Substrate.cl", "Interaction_Substrate"
        )

    def setArgs(self):
        self.knl.set_args(
            self.simulation.force.buf,
            self.simulation.points.buf,
            ctypes.c_double(self.wallPosition),
            ctypes.c_double(self.forceConst),
        )
