import sys

import Simulation as s

SETUP = sys.argv[2]
testSim = s.Simulation(sys.argv[1])

if SETUP == "TL":
    print("setting setup to TIPLESS")
    testSim.setInteractionTiltedPlane(isSpherical=False)
elif SETUP == "SP":
    print("setting setup to SPHERICAL")
    testSim.setInteractionSphere(isSpherical=False)
elif SETUP == "ST":
    print("setting setup to SHARP TIP")
    testSim.setInteractionTipIntegral(isSpherical=False)
else:
    print("define the setup type: TL/SP/ST")
    exit()
testSim.setInteractionHalfPlane()
testSim.setInteractionNeoHookean()
testSim.setInteractionPointZeroForce()
testSim.setInteractionVelocityVerlet()
testSim.run(writePressure=True, writeStress=True, writeVolumeChange=False)
