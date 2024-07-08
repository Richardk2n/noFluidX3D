import Simulation as s
import sys

def runSim(paramFile):
    sim = s.Simulation(paramFile)
    sim.setInteractionSphere()
    sim.setInteractionHalfPlane()
    sim.setInteractionSecondOrderNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

runSim(sys.argv[1])

