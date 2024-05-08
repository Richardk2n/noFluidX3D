import Simulation as s
import sys
import json

def heteroFunc(x):
    return 1-0.5*x**2

def runSim(paramFile):
    sim = s.Simulation(paramFile)
    sim.setInteractionSphere()
    sim.setInteractionHalfPlane()
    sim.setInteractionNeoHookean(heteroFunction=heteroFunc)
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

runSim(sys.argv[1])
