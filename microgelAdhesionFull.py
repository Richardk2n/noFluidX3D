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

#adhesionConstants = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
#adhesionConstants = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
adhesionConstants = [0.05, 0.06]
#indices = [0,1,2,3,4,5,6,7,8]
indices = [5,6]
configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
for i in range(len(adhesionConstants)):
    parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/Adhesion/"+title+"_adhesionConstant_%.3f/" % adhesionConstants[i]
    if SETUP == "ST":
        parameters["CELL"]["CheckPointVTK"] = "AdhesionVTKs/mesh_relaxedSofter_ST_%d.vtk" % indices[i]
    else:
        parameters["CELL"]["CheckPointVTK"] = "AdhesionVTKs/mesh_relaxedSofter_SP_%d.vtk" % indices[i]
    parameters["AdhesionConstant"] = adhesionConstants[i]
    paramFile = "AdhesionConfigs/config_" + title + "_adhesionConstant_%.3f.json" % adhesionConstants[i]
    with open(paramFile, "w") as f:
        f.write(json.dumps(parameters))
    paramFiles.append(paramFile)
    
def runSim(paramFile):
    sim = s.Simulation(paramFile)
    sim.setInteractionAdhesivePlaneNonDimensional(isSpherical=False)
    sim.initCheckPointVTK()
    if SETUP == "TL":
        print("setting setup to TIPLESS")
        sim.setInteractionTiltedPlane(isSpherical=False)
    elif SETUP == "SP":
        print("setting setup to SPHERICAL")
        sim.setInteractionSphere(isSpherical=False)
    elif SETUP == "ST":
        print("setting setup to SHARP TIP")
        sim.setInteractionTipIntegral(isSpherical=False)
    else:
        print("define the setup type: TL/SP/ST")
        exit()
    sim.setInteractionNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet(fixTopBottom=True)
    sim.run()

# Run all the simulations, one after the other
for i in range(len(paramFiles)):
    print(f"starting simulation of configfile {paramFiles[i]}")
    runSim(paramFiles[i])
