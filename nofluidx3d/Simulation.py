# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 14:21:58 2025

@author: Richard Kellnberger
"""

import json
import os
import time

import numpy as np

from nofluidx3d import TetraCell as tc
from nofluidx3d.interactions import MooneyRivlin, PlaneAFM, VelocityVerlet
from nofluidx3d.openCL import getCommandQueue, initializeOpenCLObjects
from nofluidx3d.util.Bridge import Bridge
from nofluidx3d.util.IO import writeVTK


class Simulation:
    def __init__(self, con):
        configfile = open(con)
        self.parameters = json.load(configfile)

        initializeOpenCLObjects(self.parameters["gpu"])

        # Calculate all conversion constants
        rho = 1.0
        self.rho0 = self.parameters["RhoSI"] / rho
        self.L0 = self.parameters["CELL"]["RadiusSI"] / self.parameters["CELL"]["RadiusSIM"]

        E = 1e-4
        self.p0 = self.parameters["CELL"]["YoungsModulusSI"] / E

        self.V0 = np.sqrt(self.p0 / self.rho0)
        self.T0 = self.L0 / self.V0

        print(f"V0={self.V0}")
        print(f"T0={self.T0}")

        self.Nsteps = int(round((self.parameters["SimulationTimeSI"]) / self.T0))
        self.VTKInterval = int(round(self.Nsteps / self.parameters["VTKFramesTotal"]))
        self.vtkfilespath = self.parameters["OutputDir"] + "vtkfiles/frame"
        configfile.close()

        try:
            os.makedirs(self.parameters["OutputDir"] + "vtkfiles")
        except:
            pass

        self.cell = tc.TetraCell()
        self.cell.initFromVTK(
            self.parameters["CELL"]["InitVTK"],
            radius=self.parameters["CELL"]["RadiusSIM"],
            recenter=True,
        )

        self.numPoints = self.cell.points[:, 0].size
        self.numTetra = self.cell.tetras[:, 0].size

        print(
            f"Particle mesh consists of {self.numPoints} vertices and {self.numTetra} tetrahedrons"
        )
        self.interactions = []
        self.time = 0

        # Init GPU Buffers that are (potentially) shared by multiple kernels and also potentially read from again

        self.points = Bridge(
            self.cell.points.reshape(3 * self.numPoints, order="F"), datatype=np.float64
        )
        self.tetras = Bridge(
            self.cell.tetras.reshape(4 * self.numTetra, order="F"), datatype=np.uint32
        )
        self.force = Bridge(np.zeros(self.numPoints * 3), datatype=np.float64)
        self.vonMises = Bridge(np.zeros(self.numTetra), datatype=np.float64)
        self.pressure = Bridge(np.zeros(self.numTetra), datatype=np.float64)

        # Documented values for analysis, add functions that give the current values for recording
        # They are called after writeVTK reads positions from VRAM, so dont do it again in the functions
        self.recordedQuantities = []

        # Example: time
        def currentTime():
            return self.time

        self.recordedQuantities.append(currentTime)

    def register(self, interaction):
        interaction.register(self)
        self.interactions.append(interaction)

    def setInteractionPlaneAFM(self, record=True):
        def topWallFunc(time):
            if time <= self.parameters["MoveTimeSI"] / self.T0:
                return (
                    self.parameters["CELL"]["RadiusSIM"]
                    + self.parameters["InitialDistance"]
                    - self.parameters["VelocitySI"] / self.V0 * time
                )
            else:
                return (
                    self.parameters["CELL"]["RadiusSIM"]
                    + self.parameters["InitialDistance"]
                    - self.parameters["VelocitySI"]
                    / self.V0
                    * self.parameters["MoveTimeSI"]
                    / self.T0
                )

        def bottomWallFunc(time):
            return -(self.parameters["CELL"]["RadiusSIM"] + self.parameters["InitialDistance"])

        forceConst = self.parameters["PotentialForceConst"]
        interAFM = PlaneAFM(
            topWallFunc,
            bottomWallFunc,
            forceConst,
        )
        self.register(interAFM)

        # this records the distance the sphere has travelled
        def indentationSI():
            return (topWallFunc(0) - topWallFunc(self.time)) * self.L0

        # this records the force exerted onto the sphere by the cell
        def forceSI():
            dis = topWallFunc(self.time) - self.cell.mesh.points[:, 1]
            force_abs = np.exp(-forceConst * dis)
            force = force_abs * (self.p0 * self.L0**2)
            return sum(force)

        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)

    def setInteractionMooneyRivlin(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.p0
        mooneyRivlinRatio = self.parameters["CELL"]["MooneyRivlinRatio"]
        interMR = MooneyRivlin(
            self.cell,
            youngsModulus,
            poissonRatio,
            mooneyRivlinRatio,
        )
        self.register(interMR)

    # Velocity Verlet Interaction, always add time integration interactions as last element in the list so that its kernel is queued last!
    def setInteractionVelocityVerlet(self, fixTopBottom=False):
        interVV = VelocityVerlet(self.numPoints, fixTopBottom)
        self.register(interVV)

    def timeStep(self):
        for interaction in self.interactions:
            interaction.beforeTimeStep(self.time)
        for interaction in self.interactions:
            interaction.enqueue()
        self.time += 1

    def run(self, writeStress=True, writePressure=True, writeVolumeChange=False):
        print("\n\n")
        print("+------------------------------------+")
        print("|         Starting Simulation        |")
        print("+------------------------------------+")
        print("\n\n")
        starttimes = []
        endtimes = []
        numberCycles = self.parameters["VTKFramesTotal"]
        dataPoints = None

        writeVTK(
            self,
            writeStress=writeStress,
            writePressure=writePressure,
            writeVolumeChange=writeVolumeChange,
        )

        for i in range(numberCycles):
            anatime1 = time.time()
            newDataPoints = []
            for quant in self.recordedQuantities:
                newDataPoints.append(quant())
            if self.time == 0:  # if first frame
                dataPoints = np.array(newDataPoints)
            else:
                dataPoints = np.vstack((dataPoints, newDataPoints))
            np.savetxt(self.parameters["OutputDir"] + "simdata.dat", dataPoints)
            anatime2 = time.time()
            starttimes.append(time.time())
            for t in range(self.VTKInterval):
                self.timeStep()
            getCommandQueue().finish()
            writeVTK(
                self,
                writeStress=writeStress,
                writePressure=writePressure,
                writeVolumeChange=writeVolumeChange,
            )
            endtimes.append(time.time())
            cycleTime = np.average(np.array(endtimes) - np.array(starttimes))
            vtktime = anatime2 - anatime1
            print(
                f"| Average # of simulation steps per second: {np.round(self.VTKInterval / cycleTime, 1)} | VTK cycle took {np.round(vtktime,1)} seconds | estimate of time remaining: {np.round(((numberCycles - i) * (cycleTime + vtktime))/ 60, 1)} min",
                end="\r",
            )
        print("\n\n")
        print("+------------------------------------+")
        print("|  Simulation finished successfully  |")
        print("+------------------------------------+")
