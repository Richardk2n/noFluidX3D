# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 14:21:58 2025

@author: Richard Kellnberger
"""

import json
import os
import time

try:
    from warnings import deprecated  # Python 3.13
except ImportError:

    def deprecated(message: str):  # TODO write something useful
        def decorator(fun):
            return fun

        return decorator


import numpy as np

from nofluidx3d import TetraCell as tc
from nofluidx3d.interactions import (
    MooneyRivlin,
    Plane,
    PlaneAFM,
    SaintVenantKirchhoff,
    Sphere,
    Substrate,
    VelocityVerlet,
)
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

        E = 1e-2
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

    def setInteractionSubstrate(self):
        wallPosition = -(self.parameters["CELL"]["RadiusSIM"] + self.parameters["InitialDistance"])

        forceConst = 0.1  # self.parameters["PotentialForceConst"]
        interSubstrate = Substrate(
            wallPosition,
            forceConst,
        )
        self.register(interSubstrate)

    def setInteractionPlane(self, record=True):
        def wallFunc(time):
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

        forceConst = 0.1  # self.parameters["PotentialForceConst"]
        interPlane = Plane(
            wallFunc,
            forceConst,
        )
        self.register(interPlane)

        # this records the distance the sphere has travelled
        def indentationSI():
            return (wallFunc(0) - wallFunc(self.time)) * self.L0

        # this records the force exerted onto the sphere by the cell
        def forceSI():
            dis = np.max(
                [self.cell.mesh.points[:, 1] - wallFunc(self.time), np.zeros(self.numPoints)], 0
            )
            print(f"Max dis: {np.max(dis)}")
            force_abs = forceConst * dis
            force = force_abs * (self.p0 * self.L0**2)
            return sum(force)

        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)

    def setInteractionSphere(self, record=True, isSpherical=True):
        radius = self.parameters["SphereRadiusSI"] / self.L0
        if isSpherical:
            sphereStartingPos = (
                self.parameters["CELL"]["RadiusSIM"] + radius + self.parameters["InitialDistance"]
            )
        else:
            sphereStartingPos = (
                self.cell.mesh.points[3][1] + radius + self.parameters["InitialDistance"]
            )
        print(
            f"DEBUG: sphereStartingPos = {sphereStartingPos}, lowest point = {sphereStartingPos - radius}"
        )

        def sphereFunc(time):
            if time <= self.parameters["MoveTimeSI"] / self.T0:
                return sphereStartingPos - self.parameters["VelocitySI"] / self.V0 * time
            else:
                return (
                    sphereStartingPos
                    - self.parameters["VelocitySI"]
                    / self.V0
                    * self.parameters["MoveTimeSI"]
                    / self.T0
                )

        forceConst = 0.1  # self.parameters["PotentialForceConst"]
        interSphere = Sphere(
            radius,
            sphereFunc,
            forceConst,
        )
        self.register(interSphere)

        # this records the distance the sphere has travelled
        def indentationSI():
            return (sphereFunc(0) - sphereFunc(self.time)) * self.L0

        # this records the force exerted onto the sphere by the cell
        def forceSI():
            sphereposvec = np.array([0, sphereFunc(self.time), 0])
            length = np.linalg.norm(sphereposvec - self.cell.mesh.points, axis=-1)
            normals = (self.cell.mesh.points - sphereposvec) / np.stack(
                (length, length, length), axis=-1
            )
            dis = np.max([radius - length, np.zeros(self.numPoints)], 0)
            print(f"Max dis: {np.max(dis)}")
            force_abs = forceConst * dis
            force = (
                np.stack((force_abs, force_abs, force_abs), axis=-1)
                * normals
                * (self.p0 * self.L0**2)
            )
            return -sum(force[..., 1])

        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)

    @deprecated("Switch to Plane + Substrate")
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

        forceConst = 0.1  # self.parameters["PotentialForceConst"]
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
            dis = np.max(
                [self.cell.mesh.points[:, 1] - topWallFunc(self.time), np.zeros(self.numPoints)], 0
            )
            print(f"Max dis: {np.max(dis)}")
            force_abs = forceConst * dis
            force = force_abs * (self.p0 * self.L0**2)
            return sum(force)

        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)

    def setInteractionSaintVenantKirchhoff(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.p0
        interLE = SaintVenantKirchhoff(
            self.cell,
            youngsModulus,
            poissonRatio,
        )
        self.register(interLE)

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
        print(f"VTKInterval: {self.VTKInterval}")
        starttimes = []
        endtimes = []
        numberCycles = self.parameters["VTKFramesTotal"]

        writeVTK(
            self,
            writeStress=writeStress,
            writePressure=writePressure,
            writeVolumeChange=writeVolumeChange,
        )
        newDataPoints = []
        for quant in self.recordedQuantities:
            newDataPoints.append(quant())
        dataPoints = np.array(newDataPoints)
        np.savetxt(self.parameters["OutputDir"] + "simdata.dat", dataPoints)

        for i in range(numberCycles):
            starttimes.append(time.time())
            for t in range(self.VTKInterval):
                self.timeStep()
            getCommandQueue().finish()
            anatime1 = time.time()
            writeVTK(
                self,
                writeStress=writeStress,
                writePressure=writePressure,
                writeVolumeChange=writeVolumeChange,
            )
            newDataPoints = []
            for quant in self.recordedQuantities:
                newDataPoints.append(quant())
            dataPoints = np.vstack((dataPoints, newDataPoints))
            np.savetxt(self.parameters["OutputDir"] + "simdata.dat", dataPoints)
            anatime2 = time.time()
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
