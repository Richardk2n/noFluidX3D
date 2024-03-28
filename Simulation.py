import json
import TetraCell as tc
import numpy as np
import pyopencl as cl
import Interactions as inter
from pyclParams import ctx, queue, mf
import time

class Bridge:
    def __init__(self, startarray, datatype):
        self.arrayC = startarray.astype(datatype)
        self.buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.arrayC)
        
    def readFromGPU(self):
        cl.enqueue_copy(queue, self.arrayC, self.buf)
        return self.arrayC

class Simulation:
    def __init__(self, con):
        configfile = open(con)
        self.parameters = json.load(configfile)
        # Calculate all conversion constants
        rho = 1.0
        self.rho0 = self.parameters["RhoSI"] / rho
        muSI = self.parameters["MuFluidSI"] / self.parameters["ReynoldScaling"]
        nuSI = muSI / self.parameters["RhoSI"]
        nu = 1.0 / 6.0 * self.parameters["TimeStepScaling"]
        self.nu0 = nuSI / nu
        self.L0 = self.parameters["CELL"]["RadiusSI"] / self.parameters["CELL"]["RadiusSIM"]
        self.E0 = self.rho0 * self.nu0**2 * self.L0
        self.T0 = np.sqrt(self.rho0 / self.E0 * self.L0**5)
        self.V0 = self.L0 / self.T0
        self.p0 = self.rho0 * self.V0**2

        self.Nsteps = int(round((self.parameters["SimulationTimeSI"]) / self.T0))
        self.VTKInterval = int(round(self.Nsteps / self.parameters["VTKFramesTotal"]))
        self.vtkfilespath = self.parameters["OutputDir"] + "vtkfiles/frame"
        configfile.close()
        
        
        self.cell = tc.TetraCell()
        self.cell.initFromVTK(self.parameters["CELL"]["InitVTK"])
        
        self.numPoints = self.cell.points[:,0].size
        self.numTetra = self.cell.tetras[:,0].size
        
        self.interactions = []
        self.time = 0
        
        # Init GPU Buffers that are (potentially) shared by multiple kernels and also potentially read from again

        self.points = Bridge(self.cell.points.reshape(3*self.numPoints, order='F'), datatype=np.float64)
        self.tetras = Bridge(self.cell.tetras.reshape(4*self.numTetra, order='F'), datatype=np.uint32)
        self.force = Bridge(np.zeros(self.numPoints*3), datatype=np.float64)
        
    def setInteractionSphere(self):
        radius = self.parameters["SphereRadiusSI"] / self.L0
        def sphereFunc(time):
            if(time <= self.parameters["MoveTimeSI"] / self.T0):
                return (self.parameters["CELL"]["RadiusSIM"]
                        + self.parameters["SphereRadiusSI"]
                        / self.L0
                        + self.parameters["InitialDistance"]
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * time)
            else:
                return (self.parameters["CELL"]["RadiusSIM"]
                        + self.parameters["SphereRadiusSI"]
                        / self.L0
                        + self.parameters["InitialDistance"]
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * self.parameters["MoveTimeSI"]
                        / self.T0)
        forceConst = self.parameters["PotentialForceConst"]
        interSphere = inter.InteractionSphere(self.numPoints, self.numTetra, self.force.buf, self.points.buf, radius, sphereFunc, forceConst)
        self.interactions.append(interSphere)
    
    def setInteractionTip(self):
        radius = self.parameters["SphereRadiusSI"] / self.L0
        def sphereFunc(time):
            if(time <= self.parameters["MoveTimeSI"] / self.T0):
                return (self.parameters["CELL"]["RadiusSIM"]
                        + self.parameters["SphereRadiusSI"]
                        / self.L0
                        + self.parameters["InitialDistance"]
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * time)
            else:
                return (self.parameters["CELL"]["RadiusSIM"]
                        + self.parameters["SphereRadiusSI"]
                        / self.L0
                        + self.parameters["InitialDistance"]
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * self.parameters["MoveTimeSI"]
                        / self.T0)
        forceConst = self.parameters["PotentialForceConst"]
        halfAngle = self.parameters["TipHalfAngleDeg"] / 360.0 * 2*np.pi
        interTip = inter.InteractionTip(self.numPoints, self.numTetra, self.force.buf, self.points.buf, radius, halfAngle, sphereFunc, forceConst)
        self.interactions.append(interTip)
    
    def setInteractionHalfPlane(self, isBelow=True):
        def wallFunc(time):
            return - self.parameters["CELL"]["RadiusSIM"] - self.parameters["InitialDistance"]
        forceConst = self.parameters["PotentialForceConst"]
        interPlane = inter.InteractionHalfPlane(self.numPoints, self.numTetra, self.force.buf, self.points.buf, wallFunc, forceConst, isBelow)
        self.interactions.append(interPlane)

    def setInteractionLinearElastic(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 

        interLE = inter.InteractionLinearElastic(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, youngsModulus, poissonRatio)
        self.interactions.append(interLE)
    
    def setInteractionCOMForce(self):
        springConst = self.parameters["COMSpringConst"]
        self.com = Bridge(np.zeros(2), datatype=np.float64)
        interCalcCOM = inter.InteractionCalcCOM(self.numPoints, self.numTetra, self.points.buf, self.com.buf)
        interCOMForce = inter.InteractionCOMForce(self.numPoints, self.numTetra, self.force.buf, self.points.buf, springConst, self.com.buf)
        self.interactions.append(interCalcCOM)
        self.interactions.append(interCOMForce)

    # A less accurate but also faster alternative to the COMForce
    def setInteractionPointZeroForce(self):
        springConst = self.parameters["COMSpringConst"]
        interPZF = inter.InteractionPointZeroForce(self.numPoints, self.numTetra, self.force.buf, self.points.buf, springConst)
        self.interactions.append(interPZF)
        
    # Velocity Verlet Interaction, always add time integration interactions as last element in the list so that its kernel is queued last!
    def setInteractionVelocityVerlet(self):
        interVV = inter.InteractionVelocityVerlet(self.numPoints, self.numTetra, self.force.buf, self.points.buf)
        self.interactions.append(interVV)
    
    # Overdamped Interaction, always add time integration interactions as last element in the list so that its kernel is queued last!
    def setInteractionOverDamped(self):
        interOD = inter.InteractionOverDamped(self.numPoints, self.numTetra, self.force.buf, self.points.buf)
        self.interactions.append(interOD)
        
    def timeStep(self):
        for interaction in self.interactions:
            interaction.beforeTimeStep(self.time)
        for interaction in self.interactions:
            interaction.queueKernel()
        self.time += 1

    def writeVTK(self):
        # get current coordinates from VRAM and input them into the cell objects mesh
        positions = self.points.readFromGPU()
        positions = positions.reshape((3, self.numPoints)).transpose()
        self.cell.mesh.points = positions
        
        # write a vtk file named with the current timestep
        self.cell.mesh.save(self.vtkfilespath+f"{self.time}.vtk")

    def run(self):
        times = [time.time()]
        numberCycles = self.parameters["VTKFramesTotal"]
        for i in range(numberCycles):
            self.writeVTK()
            for t in range(self.VTKInterval):
                self.timeStep()
            times.append(time.time())
            cycleTime = np.average(np.array(times)[1:] - np.array(times)[:-1])
            print(f"steps per second: {self.VTKInterval / cycleTime} \t time remaining: {(numberCycles - i) * cycleTime}", end="\r")

















