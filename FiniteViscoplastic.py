import Simulation as s
import sys
testSim = s.Simulation(sys.argv[1])

testSim.setInteractionSphere(isSpherical=False)
testSim.setInteractionHalfPlane()
testSim.setInteractionFiniteStrainViscoplastic()
testSim.setInteractionPointZeroForce()
testSim.setInteractionVelocityVerlet()

testSim.run(writePressure=True, writeStress=True, writeVolumeChange=False)
