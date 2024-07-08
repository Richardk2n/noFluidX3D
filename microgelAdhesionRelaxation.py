import Simulation as s
import sys

#SETUP = sys.argv[2]
SETUP = None

testSim = s.Simulation(sys.argv[1])
testSim.initCheckPointVTK()
if SETUP == "TL":
    print("setting setup to TIPLESS")
    testSim.setInteractionTiltedPlane(isSpherical=False)
elif SETUP == "SP":
    print("setting setup to SPHERICAL")
    testSim.setInteractionSphere(isSpherical=False)
elif SETUP == "ST":
    print("setting setup to SHARP TIP")
    testSim.setInteractionTipIntegral(isSpherical=False)
testSim.setInteractionAdhesivePlaneSurfaceIntegral()
testSim.setInteractionNeoHookean()
testSim.setInteractionPointZeroForce()
testSim.setInteractionVelocityVerlet()
testSim.run(writePressure=True, writeStress=True, writeVolumeChange=False)
