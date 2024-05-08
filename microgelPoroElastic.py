import Simulation as s
import sys
import json

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelPoroElastic.py [path/to/referenceConfig] [title]

referenceConfig = sys.argv[1]
title = sys.argv[2]
TL = sys.argv[3]

#poroFactors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
poroFactors = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# setup one config file for each poisson ratio
    
configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
for poroFactor in poroFactors:
    parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/poroElastic/" + title +"/poroFactor_"+str(poroFactor)+"/"
    parameters["CELL"]["PoroElasticFactor"] = poroFactor
    paramFile = "poroConfigs/config_" + title +"_poroFactor_%.2f.json" % poroFactor
    with open(paramFile, "w") as f:
        f.write(json.dumps(parameters))
    paramFiles.append(paramFile)
    
def runSim(paramFile):
    sim = s.Simulation(paramFile)
    if TL == "TL":
        print("setting setup to Tipless")
        sim.setInteractionTiltedPlane()
    else:
        print("setting setup to Spherical")
        sim.setInteractionSphere()
    sim.setInteractionHalfPlane()
    sim.setInteractionLinearPoroElastic()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for paramFile in paramFiles:
    print(f"starting simulation of configfile {paramFile}")
    runSim(paramFile)
