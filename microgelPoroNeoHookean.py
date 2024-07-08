import Simulation as s
import sys
import json

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelPoroNeoHookean.py [path/to/referenceConfig] [title] [setup type SP/TL/ST]

referenceConfig = sys.argv[1]
title = sys.argv[2]
SETUP = sys.argv[3]

volumeFractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
#volumeFractions = [0.5, 0.7, 0.9]

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
    if SETUP == "TL":
        print("setting setup to TIPLESS")
        sim.setInteractionTiltedPlane()
    elif SETUP == "SP":
        print("setting setup to SPHERICAL")
        sim.setInteractionSphere()
    elif SETUP == "ST":
        print("setting setup to SHARP TIP")
        sim.setInteractionTipIntegral()
    else:
        print("define the setup type: TL/SP/ST")
        exit()
    sim.setInteractionHalfPlane()
    sim.setInteractionPoroNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for paramFile in paramFiles:
    print(f"starting simulation of configfile {paramFile}")
    runSim(paramFile)
