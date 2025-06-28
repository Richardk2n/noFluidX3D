# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Sun Jun 15 13:45:41 2025

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


class Tetra(Interaction):
    def __init__(
        self,
        referenceCell,
        youngsModulus,
        poissonRatio,
    ):
        self.referenceCell = referenceCell
        self.youngsModulus = youngsModulus
        self.poissonRatio = poissonRatio

    def createBuffers(self):
        numTetra = self.simulation.numTetra

        # create additional buffers for referenceEdgeVectors, referenceVolumes
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
            def_pointCount=self.simulation.numPoints,
            def_tetraCount=self.simulation.numTetra,
            def_tetraYoungsModulus=self.youngsModulus,
            def_tetraPoissonRatio=self.poissonRatio,
            TETRA_SAINT_VENANT_KIRCHHOFF_BIOT=True,
            TETRA_SAINT_VENANT_KIRCHHOFF=False,
            TETRA_NEO_HOOKEAN=False,
        )
        self.knl = KernelBuilder.build(
            kernels / "Interactions" / "tetra.cl",
            "calcTetraForce",
        )

    def setArgs(self):
        self.knl.set_args(
            self.simulation.force.buf,
            self.simulation.points.buf,
            self.simulation.tetras.buf,
            self.edgeVectorsB,
            self.volumesB,
        )
