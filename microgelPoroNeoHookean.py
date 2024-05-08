import Simulation as s
import sys
import json

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelPoroElastic.py [path/to/referenceConfig] [title]

referenceConfig = sys.argv[1]
title = sys.argv[2]
TL = sys.argv[3]
#volumeFractions = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
volumeFractions = [0.5, 0.7, 0.9]

# setup one config file for each poisson ratio
    
configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
for volumeFraction in volumeFractions:
    parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/poroNeoHookean/" + title +"/volumeFraction_"+str(volumeFraction)+"/"
    parameters["CELL"]["PorousVolumeFraction"] = volumeFraction
    paramFile = "poroConfigs/config_" + title +"_volumeFraction_%.2f.json" % volumeFraction
    with open(paramFile, "w") as f:
        f.write(json.dumps(parameters))
    paramFiles.append(paramFile)
    
def runSim(paramFile):
    sim = s.Simulation(paramFile)
    if TL == "TL":
        print("setting setup to TIPLESS")
        sim.setInteractionTiltedPlane()
    else:
        print("setting setup to SPHERICAL")
        sim.setInteractionSphere()
    sim.setInteractionHalfPlane()
    sim.setInteractionPoroNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for paramFile in paramFiles:
    print(f"starting simulation of configfile {paramFile}")
    runSim(paramFile)
