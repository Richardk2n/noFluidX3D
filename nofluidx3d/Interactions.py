import ctypes

import numpy as np
import pyopencl as cl

from nofluidx3d.pyclParams import ctx, mf, queue

fluidx3d_lib = "../../fluidx3d_lib"


def readKernel(path, numPoints, numTetra, point3fix=True, numTriangle=None):
    with open(path, "r") as f:
        kernelstring = f.read()
    with open("kernels/atomicAdd.cl", "r") as f:
        atomicAdd = f.read()
    returnstring = ""
    if (
        point3fix
    ):  # special exception for the time evolution kernels. Setting this to true makes point 3 only move in y direction
        returnstring += "#define POINT3FIX\n"
    if numTriangle:
        returnstring += f"#define INSERT_NUM_TRIANGLES {numTriangle}\n"
    returnstring += (
        f"#define INSERT_NUM_POINTS {numPoints}\n #define INSERT_NUM_TETRAS {numTetra} \n typedef double ibmPrecisionFloat;\n typedef double3 ibmPrecisionFloat3;\n"
        + atomicAdd
        + kernelstring
    )
    return returnstring


class Interaction:
    def __init__(self, scope):
        self.scope = scope
        self.knl = None

    def beforeTimeStep(self, globalTime):
        pass

    def queueKernel(self):
        ev = cl.enqueue_nd_range_kernel(queue, self.knl, (self.scope,), None)


class InteractionSphere(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, radius, sphereFunc, forceConst):
        Interaction.__init__(self, numPoints)
        self.radius = radius
        self.sphereFunc = sphereFunc
        self.forceConst = forceConst
        self.prg = cl.Program(
            ctx,
            readKernel(f"{fluidx3d_lib}/InteractionSphere.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.Interaction_Sphere
        self.forceB = forceB
        self.pointsB = pointsB
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.sphereFunc(0)),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.sphereFunc(globalTime)),
            ctypes.c_double(self.forceConst),
        )


class InteractionTiltedPlane(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, angle, positionFunc, forceConst):
        Interaction.__init__(self, numPoints)
        self.angle = angle
        self.positionFunc = positionFunc
        self.forceConst = forceConst
        self.prg = cl.Program(
            ctx,
            readKernel(
                f"{fluidx3d_lib}/InteractionTiltedPlane.cl",
                numPoints,
                numTetra,
            ),
        ).build()
        self.knl = self.prg.Interaction_TiltedPlane
        self.forceB = forceB
        self.pointsB = pointsB
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.angle),
            ctypes.c_double(self.positionFunc(0)),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.angle),
            ctypes.c_double(self.positionFunc(globalTime)),
            ctypes.c_double(self.forceConst),
        )


class InteractionPlaneAFM(Interaction):
    def __init__(
        self, numPoints, numTetra, forceB, pointsB, topWallFunc, bottomWallFunc, forceConst
    ):
        Interaction.__init__(self, numPoints)
        self.topWallFunc = topWallFunc
        self.bottomWallFunc = bottomWallFunc
        self.forceConst = forceConst
        self.prg = cl.Program(
            ctx,
            readKernel(f"{fluidx3d_lib}/InteractionPlaneAFM.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.Interaction_PlaneAFM
        self.forceB = forceB
        self.pointsB = pointsB
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.topWallFunc(0)),
            ctypes.c_double(self.bottomWallFunc(0)),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.topWallFunc(globalTime)),
            ctypes.c_double(self.bottomWallFunc(globalTime)),
            ctypes.c_double(self.forceConst),
        )


class InteractionLinearElastic(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        youngsModulus,
        poissonRatio,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        youngsNP = youngsModulus * np.ones(numTetra).astype(np.float64)
        poissonNP = poissonRatio * np.ones(numTetra).astype(np.float64)
        self.youngsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=youngsNP)
        self.poissonB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=poissonNP)
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionLinearElastic.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_LinearElastic
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.youngsB,
            self.poissonB,
            vonMisesB,
            pressureB,
        )


class InteractionLinearElasticDeviatoric(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        shearModulus,
        bulkModulus,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        shearNP = shearModulus * np.ones(numTetra).astype(np.float64)
        bulkNP = bulkModulus * np.ones(numTetra).astype(np.float64)
        self.shearB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearNP)
        self.bulkB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bulkNP)
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionLinearElasticDeviatoric.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_LinearElasticDeviatoric
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.shearB,
            self.bulkB,
            vonMisesB,
            pressureB,
        )


class InteractionPoroNeoHookean(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        shearModulus,
        volumeFraction,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        # create additional buffers
        shearNP = shearModulus * np.ones(numTetra).astype(np.float64)
        volumeFractionNP = volumeFraction * np.ones(numTetra).astype(np.float64)
        self.shearB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearNP)
        self.volumeFractionB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=volumeFractionNP
        )
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionPoroNeoHookean.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_PoroNeoHookean
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.shearB,
            self.volumeFractionB,
            vonMisesB,
            pressureB,
        )


class InteractionMooneyRivlin(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        shearModulus1,
        shearModulus2,
        bulkModulus,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        shearMod1NP = shearModulus1 * np.ones(numTetra).astype(np.float64)
        shearMod2NP = shearModulus2 * np.ones(numTetra).astype(np.float64)
        bulkModNP = bulkModulus * np.ones(numTetra).astype(np.float64)
        self.shearMod1B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod1NP)
        self.shearMod2B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod2NP)
        self.bulkModB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bulkModNP)
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionMooneyRivlinStress.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_MooneyRivlinStress
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.shearMod1B,
            self.shearMod2B,
            self.bulkModB,
            vonMisesB,
            pressureB,
        )


class InteractionFiniteStrainViscoplastic(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        shearModulus1,
        shearModulus2,
        bulkModulus,
        viscoplasticFlowExponent,
        viscoplasticYieldStress,
        viscoplasticFlowRate,
        hardeningThreshhold,
        hardeningExponent,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        shearMod1NP = shearModulus1 * np.ones(numTetra).astype(np.float64)
        shearMod2NP = shearModulus2 * np.ones(numTetra).astype(np.float64)
        bulkModNP = bulkModulus * np.ones(numTetra).astype(np.float64)
        flowExponentNP = viscoplasticFlowExponent * np.ones(numTetra).astype(np.float64)
        yieldStressNP = viscoplasticYieldStress * np.ones(numTetra).astype(np.float64)
        flowRateNP = viscoplasticFlowRate * np.ones(numTetra).astype(np.float64)
        hardeningThreshholdNP = hardeningThreshhold * np.ones(numTetra).astype(np.float64)
        hardeningExponentNP = hardeningExponent * np.ones(numTetra).astype(np.float64)
        accumulatedStrainNP = np.zeros(numTetra).astype(np.float64)
        plasticDeformationTensorNP = np.repeat(np.identity(3).reshape(9), numTetra).astype(
            np.float64
        )

        self.shearMod1B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod1NP)
        self.shearMod2B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod2NP)
        self.bulkModB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bulkModNP)
        self.flowExponentB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flowExponentNP)
        self.yieldStressB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yieldStressNP)
        self.flowRateB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flowRateNP)
        self.hardeningThreshholdB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hardeningThreshholdNP
        )
        self.hardeningExponentB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hardeningExponentNP
        )
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )

        self.accumulatedStrainB = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accumulatedStrainNP
        )
        self.plasticDeformationTensorB = cl.Buffer(
            ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=plasticDeformationTensorNP
        )
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx,
            readKernel(
                f"{fluidx3d_lib}/InteractionFiniteStrainViscoplastic.cl",
                numPoints,
                numTetra,
            ),
        ).build()
        self.knl = self.prg.Interaction_FiniteStrainViscoplastic
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.shearMod1B,
            self.shearMod2B,
            self.bulkModB,
            self.flowExponentB,
            self.yieldStressB,
            self.flowRateB,
            self.hardeningThreshholdB,
            self.hardeningExponentB,
            self.accumulatedStrainB,
            self.plasticDeformationTensorB,
            vonMisesB,
            pressureB,
        )


class InteractionSecondOrderNeoHookean(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        shearModulus1,
        shearModulus2,
        bulkModulus,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        shearMod1NP = shearModulus1 * np.ones(numTetra).astype(np.float64)
        shearMod2NP = shearModulus2 * np.ones(numTetra).astype(np.float64)
        bulkModNP = bulkModulus * np.ones(numTetra).astype(np.float64)
        self.shearMod1B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod1NP)
        self.shearMod2B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shearMod2NP)
        self.bulkModB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bulkModNP)
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionSecondOrderNeoHookean.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_SecondOrderNeoHookean
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.shearMod1B,
            self.shearMod2B,
            self.bulkModB,
            vonMisesB,
            pressureB,
        )


class InteractionHalfPlane(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, wallFunc, forceConst, isBelow):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.wallFunc = wallFunc
        self.forceConst = forceConst
        self.isBelow = isBelow
        self.prg = cl.Program(
            ctx,
            readKernel(f"{fluidx3d_lib}/InteractionHalfPlane.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.Interaction_HalfPlane
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.wallFunc(0)),
            ctypes.c_double(self.forceConst),
            ctypes.c_int(self.isBelow),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.wallFunc(globalTime)),
            ctypes.c_double(self.forceConst),
            ctypes.c_int(self.isBelow),
        )


class InteractionCalcCOM(Interaction):
    def __init__(self, numPoints, numTetra, pointsB, comB):
        Interaction.__init__(self, 2)
        self.pointsB = pointsB
        self.comB = comB
        self.prg = cl.Program(
            ctx,
            readKernel("kernels/CalcCOM.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.CalcCOM
        self.knl.set_args(self.pointsB, self.comB)


class InteractionCOMForce(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, springConst, comB):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.springConst = springConst
        self.comB = comB
        self.prg = cl.Program(
            ctx,
            readKernel("kernels/COMForce.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.COMForce
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.springConst), self.comB)


class InteractionPointZeroForce(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, springConst):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.springConst = springConst
        self.prg = cl.Program(
            ctx,
            readKernel("kernels/PointZeroForce.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.PointZeroForce
        self.knl.set_args(self.forceB, self.pointsB, ctypes.c_double(self.springConst))


class InteractionTip(Interaction):
    def __init__(
        self, numPoints, numTetra, forceB, pointsB, radius, halfAngle, posFunc, forceConst
    ):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.radius = radius
        self.halfAngle = halfAngle
        self.posFunc = posFunc
        self.forceConst = forceConst
        self.prg = cl.Program(
            ctx,
            readKernel(f"{fluidx3d_lib}/InteractionTip.cl", numPoints, numTetra),
        ).build()
        self.knl = self.prg.Interaction_Tip
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.halfAngle),
            ctypes.c_double(self.posFunc(0)),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.halfAngle),
            ctypes.c_double(self.posFunc(globalTime)),
            ctypes.c_double(self.forceConst),
        )


class InteractionLinearViscoelastic(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        referenceEdgeVectors,
        referenceVolumes,
        youngsModulus1,
        poissonRatio,
        weights,
        relaxationTimes,
        vonMisesB,
        pressureB,
    ):
        Interaction.__init__(self, numTetra)
        # create additional buffers for referenceEdgeVectors, referenceVolumes, youngsModulus and poissonratio
        youngsModulusNP = youngsModulus * np.ones(numTetra).astype(np.float64)
        poissonRatioNP = poissonRatio * np.ones(numTetra).astype(np.float64)
        if len(weights) != len(relaxationTimes):
            print("weights and relaxationTimes must be of equal number!")
            exit()
        weightsNP = np.array(weights * numTetra).astype(
            np.float64
        )  # [a,b,c]*3  turns into [a,b,c,a,b,c,a,b,c]
        relaxationTimesNP = np.array(relaxationTimes * numTetra).astype(np.float64)

        self.youngsModB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=youngsModulusNP)
        self.poissonRatioB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=poissonRatioNP)
        edgeVectorsNP = referenceEdgeVectors.reshape(9 * numTetra, order="F").astype(np.float64)
        self.edgeVectorsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edgeVectorsNP)
        self.volumesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=referenceVolumes.astype(np.float64)
        )
        self.weightsB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weightsNP)
        self.relaxationsTimesB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=relaxationTimesNP
        )
        self.deviatoricStressTensorsTransient = cl.Buffer(
            ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.zeros(9 * len(weights) * numTetra).astype(np.float64),
        )
        self.oldDeviatoricStrainTensor = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.zeros(9 * numTetra).astype(np.float64)
        )

        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionLinearViscoelastic.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_LinearViscoelastic
        self.knl.set_args(
            forceB,
            pointsB,
            tetraB,
            self.edgeVectorsB,
            self.volumesB,
            self.youngsModB,
            self.poissonRatioB,
            self.weightsB,
            self.relaxationsTimesB,
            self.deviatoricStressTensorsTransientB,
            self.oldDeviatoricStrainTensor,
            ctypes.c_int(len(weights)),
            vonMisesB,
            pressureB,
        )


class InteractionSphereIntegral(Interaction):
    def __init__(
        self, numPoints, numTetra, forceB, pointsB, tetraB, radius, sphereFunc, forceConst
    ):
        Interaction.__init__(self, numTetra)
        self.radius = radius
        self.sphereFunc = sphereFunc
        self.forceConst = forceConst
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionSphereIntegral.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_SphereIntegral
        self.forceB = forceB
        self.pointsB = pointsB
        self.tetraB = tetraB
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            self.tetraB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.sphereFunc(0)),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            self.tetraB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.sphereFunc(globalTime)),
            ctypes.c_double(self.forceConst),
        )


class InteractionTipIntegral(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        tetraB,
        radius,
        halfAngle,
        posFunc,
        forceConst,
        tipX,
        tipZ,
    ):
        Interaction.__init__(self, numTetra)
        self.forceB = forceB
        self.pointsB = pointsB
        self.tetraB = tetraB
        self.radius = radius
        self.halfAngle = halfAngle
        self.posFunc = posFunc
        self.forceConst = forceConst
        self.tipX = tipX
        self.tipZ = tipZ
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionTipIntegral.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_TipIntegral
        print(tetraB)
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            self.tetraB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.halfAngle),
            ctypes.c_double(self.tipX),
            ctypes.c_double(self.posFunc(0)),
            ctypes.c_double(self.tipZ),
            ctypes.c_double(self.forceConst),
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            self.tetraB,
            ctypes.c_double(self.radius),
            ctypes.c_double(self.halfAngle),
            ctypes.c_double(self.tipX),
            ctypes.c_double(self.posFunc(globalTime)),
            ctypes.c_double(self.tipZ),
            ctypes.c_double(self.forceConst),
        )


class InteractionAdhesivePlane(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        forceB,
        pointsB,
        wallFunc,
        adhesionConst,
        distance,
        pointOnSurface,
    ):
        Interaction.__init__(self, numPoints)
        self.forceB = forceB
        self.pointsB = pointsB
        self.wallFunc = wallFunc
        self.adhesionConst = adhesionConst
        self.distance = distance
        self.pointOnSurfaceNP = pointOnSurface
        self.pointOnSurfaceB = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pointOnSurfaceNP
        )
        self.prg = cl.Program(
            ctx, readKernel("kernels/InteractionAdhesivePlane.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.Interaction_AdhesivePlane
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.wallFunc(0)),
            ctypes.c_double(self.adhesionConst),
            ctypes.c_double(self.distance),
            self.pointOnSurfaceB,
        )

    def beforeTimeStep(self, globalTime):
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            ctypes.c_double(self.wallFunc(globalTime)),
            ctypes.c_double(self.adhesionConst),
            ctypes.c_double(self.distance),
            self.pointOnSurfaceB,
        )


class InteractionAdhesivePlaneSurfaceIntegral(Interaction):
    def __init__(
        self,
        numPoints,
        numTetra,
        numTriangle,
        forceB,
        pointsB,
        triangleB,
        wallFunc,
        adhesionConst,
        distance,
    ):
        Interaction.__init__(self, numTriangle)
        self.forceB = forceB
        self.pointsB = pointsB
        self.triangleB = triangleB
        self.wallFunc = wallFunc
        self.adhesionConst = adhesionConst
        self.distance = distance
        self.prg = cl.Program(
            ctx,
            readKernel(
                "kernels/InteractionAdhesivePlaneSurfaceIntegral.cl",
                numPoints,
                numTetra,
                numTriangle=numTriangle,
            ),
        ).build()
        self.knl = self.prg.Interaction_AdhesivePlaneSurfaceIntegral
        self.knl.set_args(
            self.forceB,
            self.pointsB,
            self.triangleB,
            ctypes.c_double(self.wallFunc(0)),
            ctypes.c_double(self.adhesionConst),
            ctypes.c_double(self.distance),
        )


class InteractionVelocityVerlet(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, fixTopBottom):
        Interaction.__init__(self, numPoints)
        # create additional buffers for velocity and old forceB
        velocityNP = np.zeros(3 * numPoints).astype(np.float64)
        forceOldNP = np.zeros(3 * numPoints).astype(np.float64)
        self.velocityB = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocityNP)
        self.forceOldB = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=forceOldNP)
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx,
            readKernel("kernels/velocityVerlet.cl", numPoints, numTetra),
        ).build()
        if fixTopBottom:
            self.knl = self.prg.VelocityVerletFixedTopBottom
        else:
            self.knl = self.prg.VelocityVerlet
        self.knl.set_args(pointsB, self.velocityB, forceB, self.forceOldB)


class InteractionVelocityVerlet2(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, massesB, damping):
        Interaction.__init__(self, numPoints)
        self.damping = damping
        # create additional buffers for velocity and old forceB
        velocityNP = np.zeros(3 * numPoints).astype(np.float64)
        forceOldNP = np.zeros(3 * numPoints).astype(np.float64)
        self.velocityB = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocityNP)
        self.forceOldB = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=forceOldNP)
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx, readKernel("kernels/velocityVerlet2.cl", numPoints, numTetra)
        ).build()
        self.knl = self.prg.VelocityVerlet2
        self.knl.set_args(
            pointsB, self.velocityB, forceB, self.forceOldB, massesB, ctypes.c_double(self.damping)
        )


class InteractionOverDamped(Interaction):
    def __init__(self, numPoints, numTetra, forceB, pointsB, point3fix=True):
        Interaction.__init__(self, numPoints)
        # Compile Kernel and set arguments
        self.prg = cl.Program(
            ctx,
            readKernel(
                "kernels/OverDamped.cl",
                numPoints,
                numTetra,
                point3fix=point3fix,
            ),
        ).build()
        self.knl = self.prg.OverDamped
        self.knl.set_args(pointsB, forceB)
