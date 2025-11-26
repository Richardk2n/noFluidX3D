# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Sun Jun 15 13:45:41 2025

@author: Richard Kellnberger
"""
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

from nofluidx3d.interactions.Interaction import Interaction
from nofluidx3d.KernelBuilder import KernelBuilder
from nofluidx3d.openCL import getContext

kernels = Path(__file__).parents[1] / "kernels"


class Tetra(Interaction):

    class materialModel(Enum):
        SAINT_VENANT_KIRCHHOFF_KELLNBERGER = auto()
        SVKK = SAINT_VENANT_KIRCHHOFF_KELLNBERGER
        SAINT_VENANT_KIRCHHOFF = auto()
        SVK = SAINT_VENANT_KIRCHHOFF
        NEO_HOOKEAN = auto()
        NH = NEO_HOOKEAN
        MOONEY_RIVLIN = (auto(),)
        MR = MOONEY_RIVLIN

    def __init__(
        self,
        referenceCell,
        youngsModulus,
        poissonRatio,
        mooneyRivlinRatio=1,  # TODO make sure this is 1 for NH
        model=materialModel.NEO_HOOKEAN,
    ):
        self.referenceCell = referenceCell
        self.youngsModulus = youngsModulus
        self.poissonRatio = poissonRatio
        self.mooneyRivlinRatio = mooneyRivlinRatio
        self.model = model

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
            def_tetraMooneyRivlinRatio=self.mooneyRivlinRatio,
            TETRA_SAINT_VENANT_KIRCHHOFF_KELLNBERGER=self.model
            == self.materialModel.SAINT_VENANT_KIRCHHOFF_KELLNBERGER,
            TETRA_SAINT_VENANT_KIRCHHOFF=self.model == self.materialModel.SAINT_VENANT_KIRCHHOFF,
            TETRA_MOONEY_RIVLIN=self.model == self.materialModel.MOONEY_RIVLIN
            or self.model == self.materialModel.NEO_HOOKEAN,
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
