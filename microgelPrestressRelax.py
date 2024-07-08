import Simulation as s
import sys
import json
import numpy as np
from functools import partial

# usage:
# $ PYOPENCL _CTX='0:[gpu]' python microgelShell.py [path/to/referenceConfig] [title]


def heteroFuncYoungsBase(alpha, beta, x):
    f0 = 1.0 / (beta**3 + 3*(alpha-1)*(beta-1)**2 *(beta+1)/(np.pi**2) - 0.5*(alpha+1)*(beta**3-1))
    return f0 * ((x<beta) * 1.0
                 + (x>beta) * (1+(alpha-1)*np.sin(np.pi/2.0 * (x-beta)/(1-beta))**2)
                 )
def heteroFuncShapeBase(beta, x):
    return ((x<beta) * 0.0
            + (x>beta) * np.sin(np.pi/2.0 * (x-beta)/(1.0-beta))**2)


con = sys.argv[1]

heteroFuncYoungs = partial(heteroFuncYoungsBase, 2.0, 0.7)
heteroFuncShape = partial(heteroFuncShapeBase, 0.7)

sim = s.Simulation(con)
sim.setShellEllipsoidalReferenceState(0.9, heteroFuncShape)
sim.setInteractionNeoHookean(heteroFunction=heteroFuncYoungs)
# here checkpoint must be loaded
sim.setInteractionPointZeroForce()
sim.setInteractionVelocityVerlet()
sim.run()

