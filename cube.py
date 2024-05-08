import Simulation as s
import sys
import numpy as np

testSim = s.Simulation(sys.argv[1])
testSim.setInteractionLinearElastic()
positions = testSim.points.readFromGPU()
positions = positions.reshape((3, testSim.numPoints)).transpose()

def u1(x, gamma):
    return gamma * np.einsum("i, j -> ij", x[:, 1],  np.array([1,0,0]))

def u2(x, gamma):
    return gamma * (
        np.einsum("i, j -> ij", -x[:, 0],  np.array([1,0,0]))
        + np.einsum("i, j -> ij", -x[:, 1],  np.array([0,1,0]))
        + np.einsum("i, j -> ij", -x[:, 2],  np.array([0,0,1]))
        )

def u3(x, delta):
    return (
        (1/np.sqrt(1-delta) - 1)  * np.einsum("i, j -> ij", x[:, 0],  np.array([1,0,0]))
        + (-delta) * np.einsum("i, j -> ij", x[:, 1],  np.array([0,1,0]))
        + (1/np.sqrt(1-delta) - 1) * np.einsum("i, j -> ij", x[:, 2],  np.array([0,0,1]))
        )

testSim.writeVTK()
for i in range(100):
    gamma = i / 200.0
    positions_def = positions + u2(positions, gamma)
    testSim.points.arrayC = positions_def.reshape(3*testSim.numPoints, order='F')
    testSim.points.writeToGPU()

    testSim.timeStep()
    testSim.writeVTK()
