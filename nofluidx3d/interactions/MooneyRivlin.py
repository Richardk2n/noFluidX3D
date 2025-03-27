# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 12:32:54 2025

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


class MooneyRivlin(Interaction):
    def __init__(
        self,
        referenceCell,
        youngsModulus,
        poissonRatio,
        mooneyRivlinRatio,
    ):
        self.referenceCell = referenceCell
        self.youngsModulus = youngsModulus
        self.poissonRatio = poissonRatio
        self.mooneyRivlinRatio = mooneyRivlinRatio

    def createBuffers(self):
        numTetra = self.simulation.numTetra

        shearModulus = self.youngsModulus / (2 * (1 + self.poissonRatio))
        bulkModulus = self.youngsModulus / (3 * (1 - 2 * self.poissonRatio))

        shearModulus1 = self.mooneyRivlinRatio * shearModulus
        shearModulus2 = (1 - self.mooneyRivlinRatio) * shearModulus
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        shearMod1NP = shearModulus1 * np.ones(numTetra).astype(np.float64)
        shearMod2NP = shearModulus2 * np.ones(numTetra).astype(np.float64)
        bulkModNP = bulkModulus * np.ones(numTetra).astype(np.float64)
        self.shearMod1B = cl.Buffer(
            getContext(), mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod1NP
        )
        self.shearMod2B = cl.Buffer(
            getContext(), mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod2NP
        )
        self.bulkModB = cl.Buffer(getContext(), mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bulkModNP)
        edgeVectorsNP = self.referenceCell.edgeVectors.reshape(9 * numTetra, order="F").astype(
            np.float64
        )
        self.edgeVectorsB = cl.Buffer(
            getContext(), mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP
        )
        self.volumesB = cl.Buffer(
            getContext(),
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=self.referenceCell.volumes.astype(np.float64),
        )
        self.referenceCell = None

    def build(self):
        self.scope = self.simulation.numTetra
        KernelBuilder.define(
            INSERT_NUM_POINTS=self.simulation.numPoints, INSERT_NUM_TETRAS=self.simulation.numTetra
        )
        self.knl = KernelBuilder.build(
            kernels / "Interactions" / "MooneyRivlinStress.cl", "Interaction_MooneyRivlinStress"
        )

    def setArgs(self):
        self.knl.set_args(
            self.simulation.force.buf,
            self.simulation.points.buf,
            self.simulation.tetras.buf,
            self.edgeVectorsB,
            self.volumesB,
            self.shearMod1B,
            self.shearMod2B,
            self.bulkModB,
            self.simulation.vonMises.buf,
            self.simulation.pressure.buf,
        )
