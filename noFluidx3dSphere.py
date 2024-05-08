import Simulation as s
import sys

testSim = s.Simulation(sys.argv[1])
testSim.setInteractionSphere()
testSim.setInteractionHalfPlane()
if sys.argv[2] == "LE":
    print("Setting constitutive model to: LINEAR ELASTIC")
    testSim.setInteractionLinearElastic()
if sys.argv[2] == "NH":
    print("Setting constitutive model to: NEO HOOKEAN")
    testSim.setInteractionNeoHookean()
testSim.setInteractionPointZeroForce()
#testSim.setInteractionOverDamped(point3fix=False)
testSim.setInteractionVelocityVerlet()
testSim.run(writePressure=True, writeStress=True, writeVolumeChange=False)
