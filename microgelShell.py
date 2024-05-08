import Simulation as s
import sys
import json
import numpy as np
from functools import partial

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelShell.py [path/to/referenceConfig] [title]


def heteroFuncBase(alpha, beta, x):
    f0 = 1.0 / (beta**3 + 3*(alpha-1)*(beta-1)**2 *(beta+1)/(np.pi**2) - 0.5*(alpha+1)*(beta**3-1))
    return f0 * ((x<beta) * 1.0
                 + (x>beta) * (1+(alpha-1)*np.sin(np.pi/2.0 * (x-beta)/(1-beta))**2)
                 )

referenceConfig = sys.argv[1]
title = sys.argv[2]
TL = sys.argv[3]

#alphas = [1.5, 2.0, 3.0]
#betas = [0.7, 0.8, 0.9]

alphas = [ 1.0 ]
betas = [0.7]

configfile = open(referenceConfig)
parameters = json.load(configfile)

paramFiles = []
heteroFuncs = []
for alpha in alphas:
    for beta in betas:
        parameters["OutputDir"] = "/tp6-gekle/nas/bt709186/noFluidx3d_results/shell/"+title+"_alpha_%.1f_beta_%.1f/" % (alpha, beta)
        paramFile = "ShellConfigs/config_" + title + "_alpha_%.1f_beta_%.1f.json" % (alpha, beta)

        with open(paramFile, "w") as f:
            f.write(json.dumps(parameters))
        paramFiles.append(paramFile)
        heteroFuncs.append(partial(heteroFuncBase, alpha, beta))
    
def runSim(paramFile, heterofunc):
    sim = s.Simulation(paramFile)
    if TL == "TL":
        print("setting setup to TIPLESS")
        sim.setInteractionTiltedPlane()
    else:
        print("setting setup to SPHERICAL")
        sim.setInteractionSphere()
    sim.setInteractionHalfPlane()
    sim.setInteractionNeoHookean(heteroFunction=heterofunc)
    sim.setInteractionPointZeroForce()
    sim.setInteractionVelocityVerlet()
    sim.run()

# Run all the simulations, one after the other
for i in range(len(paramFiles)):
    print(f"starting simulation of configfile {paramFiles[i]}")
    runSim(paramFiles[i], heteroFuncs[i])
