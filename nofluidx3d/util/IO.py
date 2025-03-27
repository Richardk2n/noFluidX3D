# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Mar 27 14:41:17 2025

@author: Richard Kellnberger
"""

import numpy as np


def writeVTK(simulation, writeStress=True, writePressure=True, writeVolumeChange=False):
    # get current coordinates from VRAM and input them into the cell objects mesh
    positions = simulation.points.readFromGPU()
    positions = positions.reshape((3, simulation.numPoints)).transpose()
    if np.any(np.isnan(positions)):
        print("\n\n")
        print("+---------------------------------------------------------------------------+")
        print("| !!! Encountered NaN when reading positions from VRAM, exiting program !!! |")
        print("+---------------------------------------------------------------------------+")
        exit()
    if np.any(np.isinf(positions)):
        print("\n\n")
        print("+---------------------------------------------------------------------------+")
        print("| !!! Encountered INF when reading positions from VRAM, exiting program !!! |")
        print("+---------------------------------------------------------------------------+")
        exit()
    simulation.cell.mesh.points = positions
    if writeStress:
        stress = simulation.vonMises.readFromGPU()
        simulation.cell.addTetraData("vonMises", stress)
    if writePressure:
        pressure = -simulation.pressure.readFromGPU()
        simulation.cell.addTetraData("pressure", pressure)
    if writeVolumeChange:
        simulation.cell.calcVolumeChange()
    # write a vtk file named with the current timestep
    simulation.cell.mesh.save(simulation.vtkfilespath + f"{simulation.time}.vtk")
