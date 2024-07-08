import Simulation as s
import sys

testSim = s.Simulation(sys.argv[1])
testSim.setInteractionAdhesivePlaneNonDimensional()
testSim.setInteractionNeoHookean()
testSim.setInteractionPointZeroForce()
testSim.setInteractionVelocityVerlet(fixTopBottom=True)
testSim.run(writePressure=True, writeStress=True, writeVolumeChange=False)
