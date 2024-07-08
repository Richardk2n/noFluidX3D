import Simulation as s
import sys
import json

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelPoroElastic.py [path/to/referenceConfig] [title]

referenceConfig = sys.argv[1]
title = sys.argv[2]
SETUP = sys.argv[3]
secondOrderFactors= [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
#secondOrderFactors= [0.5, 1.0, 2.0, 5.0, 10.0]

# setup one config file for each factor
    
configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
for secondOrderFactor in secondOrderFactors:
    parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/SecondOrderNeoHookean/" + title +"/secondOrderFactor_%.2f/" % secondOrderFactor
    parameters["CELL"]["SecondOrderNeoHookeanFactor"] = secondOrderFactor
    paramFile = "SONHconfigs/config_" + title +"_secondOrderFactor_%.2f.json" % secondOrderFactor
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
    sim.setInteractionSecondOrderNeoHookean()
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for paramFile in paramFiles:
    print(f"starting simulation of configfile {paramFile}")
    runSim(paramFile)
