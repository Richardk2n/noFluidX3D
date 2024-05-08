import Simulation as s
import sys
import json

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelPoissonRatio.py [path/to/referenceConfig] [LE/NH] [title]

referenceConfig = sys.argv[1]
model = sys.argv[2]
title = sys.argv[3]
if model == "LE":
    print("Setting constitutive model to: LINEAR ELASTIC")
elif model == "NH":
    print("Setting constitutive model to: NEO HOOKEAN")
else:
    print("SET A CONSTITUTIVE MODEL!!!")
    exit()

poissons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48]

# setup one config file for each poisson ratio
    
configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
for poisson in poissons:
    parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/poissonRatio/" + title +"/poissonRatio_"+str(poisson)+"/"
    parameters["CELL"]["PoissonRatio"] = poisson
    paramFile = "poissonConfigs/config_" + title +"_poissonRatio_" + str(poisson) + ".json"
    with open(paramFile, "w") as f:
        f.write(json.dumps(parameters))
    paramFiles.append(paramFile)
    
def runSim(paramFile):
    sim = s.Simulation(paramFile)
    sim.setInteractionSphere()
    sim.setInteractionHalfPlane()
    if model == "LE":
        sim.setInteractionLinearElastic()
    if model == "NH":
        sim.setInteractionNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for paramFile in paramFiles:
    print(f"starting simulation of configfile {paramFile}")
    runSim(paramFile)
