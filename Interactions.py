import pyopencl as cl
from pyclParams import ctx, queue, mf
import ctypes
import numpy as np



def readKernel(path, numPoints, numTetra):
    with open(path, "r") as f:
        kernelstring = f.read()
    with open("atomicAdd.cl", "r") as f:
        atomicAdd = f.read()
    return f"#define INSERT_NUM_POINTS {numPoints}\n #define INSERT_NUM_TETRAS {numTetra} \n typedef double ibmPrecisionFloat;\n typedef double3 ibmPrecisionFloat3;\n" + atomicAdd + kernelstring

class Interaction:
    def __init__(self, scope):
        self.scope = scope
        self.knl = None

    def beforeTimeStep(self, globalTime):
        pass
    
    def queueKernel(self):
        ev = cl.enqueue_nd_range_kernel(queue, self.knl, (self.scope,), None)
    
        

class InteractionSphere(Interaction):
    def  __init__(self, numPoints, numTetra, forceB, pointsB, radius, sphereFunc, forceConst):
        Interaction.__init__(self, numPoints)
        self.radius = radius
        self.sphereFunc = sphereFunc
        self.forceConst=forceConst
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/fluidx3d_lib/InteractionSphere.cl", numPoints, numTetra)).build()
        self.knl = self.prg.Interaction_Sphere
        self.forceB = forceB
        self.pointsB = pointsB
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.radius), ctypes.c_double(self.sphereFunc(0)), ctypes.c_double(self.forceConst))
        
    def beforeTimeStep(self, globalTime):
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.radius), ctypes.c_double(self.sphereFunc(globalTime)), ctypes.c_double(self.forceConst))

class InteractionLinearElastic(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, tetraB, referenceEdgeVectors, referenceVolumes, youngsModulus, poissonRatio):
        Interaction.__init__(self, numTetra)
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        youngsNP = youngsModulus*np.ones(numTetra).astype(np.float64)
        poissonNP = poissonRatio*np.ones(numTetra).astype(np.float64)
        self.youngsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=youngsNP)
        self.poissonB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=poissonNP)
        
        edgeVectorsNP = referenceEdgeVectors.reshape(9*numTetra, order='F').astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64))
        # Compile Kernel and set arguments
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/fluidx3d_lib/InteractionLinearElastic.cl", numPoints, numTetra)).build()
        self.knl = self.prg.Interaction_LinearElastic
        self.knl.set_args(forceB, pointsB, tetraB, self.edgeVectorsB, self.volumesB, self.youngsB, self.poissonB)
        
class InteractionHalfPlane(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, wallFunc, forceConst, isBelow):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.wallFunc = wallFunc
        self.forceConst = forceConst
        self.isBelow = isBelow
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/fluidx3d_lib/InteractionHalfPlane.cl", numPoints, numTetra)).build()
        self.knl = self.prg.Interaction_HalfPlane
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.wallFunc(0)), ctypes.c_double(self.forceConst), ctypes.c_int(self.isBelow))

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.wallFunc(globalTime)), ctypes.c_double(self.forceConst), ctypes.c_int(self.isBelow))

class InteractionCalcCOM(Interaction):
    def __init__(self, numPoints, numTetra, pointsB, comB):
        Interaction.__init__(self, 2)
        self.pointsB = pointsB
        self.comB = comB
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/noFluidx3d/CalcCOM.cl", numPoints, numTetra)).build()
        self.knl = self.prg.CalcCOM
        self.knl.set_args(self.pointsB, self.comB)


class InteractionCOMForce(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, springConst, comB):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.springConst = springConst
        self.comB = comB
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/noFluidx3d/COMForce.cl", numPoints, numTetra)).build()
        self.knl = self.prg.COMForce
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.springConst), self.comB)

class InteractionPointZeroForce(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, springConst):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.springConst = springConst
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/noFluidx3d/PointZeroForce.cl", numPoints, numTetra)).build()
        self.knl = self.prg.PointZeroForce
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.springConst))

class InteractionTip(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, radius, halfAngle, posFunc, forceConst):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.radius = radius
        self.halfAngle  = halfAngle
        self.posFunc = posFunc
        self.forceConst = forceConst
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/fluidx3d_lib/InteractionTip.cl", numPoints, numTetra)).build()
        self.knl = self.prg.Interaction_Tip
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.radius), ctypes.c_double(self.halfAngle), ctypes.c_double(self.posFunc(0)), ctypes.c_double(self.forceConst))
        
    def beforeTimeStep(self, globalTime):
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.radius), ctypes.c_double(self.halfAngle), ctypes.c_double(self.posFunc(globalTime)), ctypes.c_double(self.forceConst))

class InteractionVelocityVerlet(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB):
        Interaction.__init__(self, numPoints)
        # create additional buffers for velocity and old forceB
        velocityNP = np.zeros(3*numPoints).astype(np.float64)
        forceOldNP = np.zeros(3*numPoints).astype(np.float64)
        self.velocityB = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocityNP)
        self.forceOldB = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=forceOldNP)
        # Compile Kernel and set arguments
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/noFluidx3d/velocityVerlet.cl", numPoints, numTetra)).build()
        self.knl = self.prg.VelocityVerlet
        self.knl.set_args(pointsB, self.velocityB, forceB, self.forceOldB)

class InteractionOverDamped(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB):
        Interaction.__init__(self, numPoints)
        # Compile Kernel and set arguments
        self.prg = cl.Program(ctx, readKernel("/tp6-gekle/nas/bt709186/noFluidx3d/OverDamped.cl", numPoints, numTetra)).build()
        self.knl = self.prg.OverDamped
        self.knl.set_args(pointsB, forceB)

        













