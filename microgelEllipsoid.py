import Simulation as s
import sys
import json
import numpy as np
from functools import partial

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelEllipsoid.py [path/to/referenceConfig] [title] [TL/SP]

referenceConfig = sys.argv[1]
title = sys.argv[2]
SETUP = sys.argv[3]


#eccentricities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
eccentricities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
for eccentricity in eccentricities:
    parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/ellipsoid/"+title+"_eccentricity_%.1f/" % eccentricity
    paramFile = "EllipsoidConfigs/config_" + title + "_eccentricity_%.1f.json" % eccentricity
    with open(paramFile, "w") as f:
        f.write(json.dumps(parameters))
    paramFiles.append(paramFile)
    
def runSim(paramFile, eccentricity):
    sim = s.Simulation(paramFile)
    sim.makeEllipsoid(eccentricity)
    if SETUP == "TL":
        print("setting setup to TIPLESS")
        sim.setInteractionTiltedPlane(isSpherical=False)
    elif SETUP == "SP":
        print("setting setup to SPHERICAL")
        sim.setInteractionSphere(isSpherical=False)
    else:
        print("define the setup type: TL/SP")
        exit()
    sim.setInteractionHalfPlane(isSpherical=False)
    sim.setInteractionNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for i in range(len(paramFiles)):
    print(f"starting simulation of configfile {paramFiles[i]}")
    runSim(paramFiles[i], eccentricities[i])
