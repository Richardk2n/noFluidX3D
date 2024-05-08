import json
import TetraCell as tc
import numpy as np
import pyopencl as cl
import Interactions as inter
from pyclParams import ctx, queue, mf
import time
import os

class Bridge:
    def __init__(self, startarray, datatype):
        self.arrayC = startarray.astype(datatype)
        self.buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.arrayC)
        
    def readFromGPU(self):
        cl.enqueue_copy(queue, self.arrayC, self.buf)
        return self.arrayC
    
    def writeToGPU(self):
        cl.enqueue_copy( queue, self.buf, self.arrayC)

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
        
        try:
            os.makedirs(self.parameters["OutputDir"] + "vtkfiles")
        except:
            pass
        
        self.cell = tc.TetraCell()
        self.cell.initFromVTK(self.parameters["CELL"]["InitVTK"], radius = self.parameters["CELL"]["RadiusSIM"])
        
        self.numPoints = self.cell.points[:,0].size
        self.numTetra = self.cell.tetras[:,0].size
        
        self.interactions = []
        self.time = 0
        
        # Init GPU Buffers that are (potentially) shared by multiple kernels and also potentially read from again

        self.points = Bridge(self.cell.points.reshape(3*self.numPoints, order='F'), datatype=np.float64)
        self.tetras = Bridge(self.cell.tetras.reshape(4*self.numTetra, order='F'), datatype=np.uint32)
        self.force = Bridge(np.zeros(self.numPoints*3), datatype=np.float64)
        self.vonMises = Bridge(np.zeros(self.numTetra), datatype=np.float64)
        self.pressure = Bridge(np.zeros(self.numTetra), datatype=np.float64)
        
        # Documented values for analysis, add functions that give the current values for recording
        # They are called after writeVTK reads positions from VRAM, so dont do it again in the functions
        self.recordedQuantities = []
        # Example: time
        def currentTime():
            return self.time
        self.recordedQuantities.append(currentTime)
       
    def makeEllipsoid(self, eccentricity):
        # Turns the spherical input mesh into an ellipsoidal disc before the simulation
        # The radius sets the major semiaxis along the x and z axis
        # the semi minor axis is the length along the y axis
        # e = sqrt(1 - b² / a² )
        # -> b/a = sqrt(1 - e²)
        scaleFactor = np.sqrt(1 - eccentricity**2)
        self.cell.scale(np.array([1, scaleFactor, 1]))
        # redefine points with new starting positions
        self.points = Bridge(self.cell.points.reshape(3*self.numPoints, order='F'), datatype=np.float64)


    def setInteractionPlaneAFM(self, record=True):
        def topWallFunc(time):
            if(time <= self.parameters["MoveTimeSI"] / self.T0):
                return (self.parameters["CELL"]["RadiusSIM"]
                        + self.parameters["InitialDistance"]
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * time)
            else:
                return (self.parameters["CELL"]["RadiusSIM"]
                        + self.parameters["InitialDistance"]
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * self.parameters["MoveTimeSI"]
                        / self.T0)
        def bottomWallFunc(time):
            return -(self.parameters["CELL"]["RadiusSIM"] + self.parameters["InitialDistance"])
        forceConst = self.parameters["PotentialForceConst"]
        interAFM = inter.InteractionPlaneAFM(self.numPoints, self.numTetra, self.force.buf, self.points.buf, topWallFunc, bottomWallFunc, forceConst)
        self.interactions.append(interAFM)
        # this records the distance the sphere has travelled
        def indentationSI():
            return (topWallFunc(0) - topWallFunc(self.time)) * self.L0
        # this records the force exerted onto the sphere by the cell
        def forceSI():
            dis = (topWallFunc(self.time) - self.cell.mesh.points[:, 1])
            force_abs = np.exp(- forceConst * dis)
            force = force_abs * (self.p0 * self.L0**2 * self.parameters["ReynoldScaling"] * self.parameters["CELL"]["YoungsScaling"])
            return sum(force)
        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)    
    
    def setInteractionSphere(self, record=True, isSpherical=True):
        radius = self.parameters["SphereRadiusSI"] / self.L0
        if isSpherical:
            sphereStartingPos = self.parameters["CELL"]["RadiusSIM"] + radius + self.parameters["InitialDistance"]
        else:
            sphereStartingPos = self.cell.mesh.points[3][1] + radius + self.parameters["InitialDistance"]
        def sphereFunc(time):
            if(time <= self.parameters["MoveTimeSI"] / self.T0):
                return (sphereStartingPos
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * time)
            else:
                return (sphereStartingPos
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * self.parameters["MoveTimeSI"]
                        / self.T0)
        forceConst = self.parameters["PotentialForceConst"]
        interSphere = inter.InteractionSphere(self.numPoints, self.numTetra, self.force.buf, self.points.buf, radius, sphereFunc, forceConst)
        self.interactions.append(interSphere)
        # this records the distance the sphere has travelled
        def indentationSI():
            return (sphereFunc(0) - sphereFunc(self.time)) * self.L0
        # this records the force exerted onto the sphere by the cell
        def forceSI():
            sphereposvec = np.array([0, sphereFunc(self.time), 0])
            length = np.linalg.norm(sphereposvec - self.cell.mesh.points, axis=-1)
            normals = ((self.cell.mesh.points - sphereposvec)/np.stack((length, length, length), axis=-1))
            dis = length - radius
            force_abs = np.exp(- forceConst * dis)
            force = np.stack((force_abs, force_abs, force_abs), axis=-1) * normals * (self.p0 * self.L0**2 * self.parameters["ReynoldScaling"] * self.parameters["CELL"]["YoungsScaling"])
            return -sum(force[..., 1])
        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)
    
    def setInteractionSphereIntegral(self, record=True, isSpherical=True):
        radius = self.parameters["SphereRadiusSI"] / self.L0
        if isSpherical:
            sphereStartingPos = self.parameters["CELL"]["RadiusSIM"] + radius + self.parameters["InitialDistance"]
        else:
            sphereStartingPos = self.cell.mesh.points[3][1] + radius + self.parameters["InitialDistance"]
        def sphereFunc(time):
            if(time <= self.parameters["MoveTimeSI"] / self.T0):
                return (sphereStartingPos
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * time)
            else:
                return (sphereStartingPos
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * self.parameters["MoveTimeSI"]
                        / self.T0)
        forceConst = self.parameters["PotentialForceConst"]
        interSphere = inter.InteractionSphere(self.numPoints, self.numTetra, self.force.buf, self.points.buf, radius, sphereFunc, forceConst)
        self.interactions.append(interSphere)
        # this records the distance the sphere has travelled
        def indentationSI():
            return (sphereFunc(0) - sphereFunc(self.time)) * self.L0
        # this records the force exerted onto the sphere by the cell

        def forceDensity(positions):
            sphereposvec = np.array([0, sphereFunc(self.time), 0])
            length = np.linalg.norm(sphereposvec - positions, axis=-1)
            normals = ((positions - sphereposvec)/np.stack((length, length, length), axis=-1))
            dis = length - radius
            force_abs = np.exp(- forceConst * dis)
            force = np.stack((force_abs, force_abs, force_abs), axis=-1) * normals 
        def forceSI():
            totalForce = np.array([0,0,0])
	        alpha = 0.58541020
	        beta = 0.13819669
            xis = np.array([[alpha, beta, beta], [beta, alpha, beta], [beta, beta, alpha], [beta, beta, beta]])
            for tetra in self.tetras:
                points = self.points[tetra]
                R1 = points[0]-points[1]
                R2 = points[0]-points[2]
                R3 = points[0]-points[3]
                volume = 1.0 / 6.0 * abs(np.dot(R1, np.cross(R2, R3)))
                integrationPoints = np.array([xis[0, 0] * points[0] + xis[0, 1] * points[1] + xis[0, 2] * points[3] + (1-xis[0, 0]-xis[0,1]-xis[0,2]) * points[4],
                                              xis[1, 0] * points[0] + xis[1, 1] * points[1] + xis[1, 2] * points[3] + (1-xis[1, 0]-xis[0,1]-xis[1,2]) * points[4],
                                              xis[2, 0] * points[0] + xis[2, 1] * points[1] + xis[2, 2] * points[3] + (1-xis[2, 0]-xis[0,1]-xis[2,2]) * points[4],
                                              xis[3, 0] * points[0] + xis[3, 1] * points[1] + xis[3, 2] * points[3] + (1-xis[3, 0]-xis[0,1]-xis[3,2]) * points[4]])
                forceTetra = volume/4.0 * np.sum(forceDensity(integrationPoints), axis=-1)
                totalForce += forceTetra
            return -totalForce[1] * (self.p0 * self.L0**2 * self.parameters["ReynoldScaling"] * self.parameters["CELL"]["YoungsScaling"])
        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)
    

    def setInteractionTip(self, record=True):
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
    
    def setInteractionTiltedPlane(self, record=True, isSpherical=True):
        planeAngle = self.parameters["PlaneAngleDeg"] / 360.0 * 2*np.pi
        if isSpherical:
            planeStartingPosition = (self.parameters["CELL"]["RadiusSIM"] + self.parameters["InitialDistance"]) / np.sin(np.pi/2.0 - planeAngle)
        else:
            majorSemiaxis = self.parameters["CELL"]["RadiusSIM"]
            minorSemiaxis = self.cell.mesh.points[3][1] # y-value of point 3 (upper point)
            thetaMin = np.arctan(majorSemiaxis / minorSemiaxis * np.tan(planeAngle)) # theta value (spherical coordinates) at which distance to plane is minimum
            planeStartingPosition = self.parameters["InitialDistance"] / np.cos(planeAngle) + majorSemiaxis * np.tan(planeAngle)*np.sin(thetaMin) + minorSemiaxis*np.cos(thetaMin)
        def positionFunc(time):
            if(time <= self.parameters["MoveTimeSI"] / self.T0):
                return (planeStartingPosition
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * time)
            else:
                return (planeStartingPosition
                        - self.parameters["VelocitySI"]
                        / self.V0
                        * self.parameters["MoveTimeSI"]
                        / self.T0)
        forceConst = self.parameters["PotentialForceConst"]
        interTP = inter.InteractionTiltedPlane(self.numPoints, self.numTetra, self.force.buf, self.points.buf, planeAngle, positionFunc, forceConst)
        self.interactions.append(interTP)

        # this records the distance the sphere has travelled
        def indentationSI():
            return (positionFunc(0) - positionFunc(self.time)) * self.L0
        # this records the force exerted onto the sphere by the cell
        def forceSI():
            posvec = np.array([0, positionFunc(self.time), 0])
            positions = self.cell.mesh.points
            normal = - np.cos(planeAngle) * np.array([0,1,0]) + np.sin(planeAngle) * np.array([1,0,0])
            dis = np.einsum("ij,j -> i", positions, normal) - np.dot(posvec, normal)
            force_abs = np.exp(- forceConst * dis)
            force = np.stack((force_abs, force_abs, force_abs), axis=-1) * normal * (self.p0 * self.L0**2 * self.parameters["ReynoldScaling"] * self.parameters["CELL"]["YoungsScaling"])
            return -sum(force[..., 1])
        if record:
            self.recordedQuantities.append(indentationSI)
            self.recordedQuantities.append(forceSI)
    
    def setInteractionHalfPlane(self, isBelow=True, isSpherical=True):
        if isSpherical:
            wallpos = - self.parameters["CELL"]["RadiusSIM"] - self.parameters["InitialDistance"]
        else:
            wallpos = - self.cell.mesh.points[3][1] - self.parameters["InitialDistance"]
        def wallFunc(time):
            return wallpos
        forceConst = self.parameters["PotentialForceConst"]
        interPlane = inter.InteractionHalfPlane(self.numPoints, self.numTetra, self.force.buf, self.points.buf, wallFunc, forceConst, isBelow)
        self.interactions.append(interPlane)

    def setInteractionLinearElastic(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 

        interLE = inter.InteractionLinearElastic(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, youngsModulus, poissonRatio, self.vonMises.buf, self.pressure.buf)
        self.interactions.append(interLE)
        
    def setInteractionLinearPoroElastic(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 
        poroElasticFactor = self.parameters["CELL"]["PoroElasticFactor"]
        shearModulus = youngsModulus / (2 * (1+poissonRatio))
        bulkModulus = (1 - poroElasticFactor) * youngsModulus / (3 * (1 - 2*poissonRatio))
        interLE = inter.InteractionLinearElasticDeviatoric(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, shearModulus, bulkModulus, self.vonMises.buf, self.pressure.buf)
        self.interactions.append(interLE)
    
    def setInteractionMooneyRivlin(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 
        mooneyRivlinRatio = self.parameters["CELL"]["MooneyRivlinRatio"]
        shearModulus = youngsModulus / (2 * (1+poissonRatio))
        bulkModulus = youngsModulus / (3 * (1 - 2*poissonRatio))
        interMR = inter.InteractionMooneyRivlin(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, mooneyRivlinRatio*shearModulus, (1 - mooneyRivlinRatio)*shearModulus, bulkModulus, self.vonMises.buf, self.pressure.buf)
        self.interactions.append(interMR)
    
    def setInteractionNeoHookean(self, heteroFunction=None):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 
        shearModulus = youngsModulus / (2 * (1+poissonRatio))
        bulkModulus = youngsModulus / (3 * (1 - 2*poissonRatio))
        # heteroFunction(R) describes the decline of stiffness at the edge of the particle and is defined for R in [0,1] with hF(R) in  [0, 1]
        if heteroFunction:
            tetraMiddles = self.cell.points[self.cell.tetras].sum(axis=1) / 4.0
            shearModulus = shearModulus * heteroFunction(np.linalg.norm(tetraMiddles, axis=-1) / self.cell.radius)
            bulkModulus  = bulkModulus  * heteroFunction(np.linalg.norm(tetraMiddles, axis=-1) / self.cell.radius)
            youngsModuli = youngsModulusSI * heteroFunction(np.linalg.norm(tetraMiddles, axis=-1) / self.cell.radius)
            self.cell.addTetraData("youngsModulus", youngsModuli) 
        interMR = inter.InteractionMooneyRivlin(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, shearModulus, 0.0*shearModulus, bulkModulus, self.vonMises.buf, self.pressure.buf)
        self.interactions.append(interMR)

    def setInteractionSecondOrderNeoHookean(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        poissonRatio = self.parameters["CELL"]["PoissonRatio"]
        secondOrderFactor = self.parameters["CELL"]["SecondOrderNeoHookeanFactor"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 
        shearModulus = youngsModulus / (2 * (1+poissonRatio))
        shearModulusSO = shearModulus * secondOrderFactor
        bulkModulus = youngsModulus / (3 * (1 - 2*poissonRatio))
        interNH = inter.InteractionSecondOrderNeoHookean(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, shearModulus, shearModulusSO, bulkModulus, self.vonMises.buf, self.pressure.buf)
        self.interactions.append(interNH)


    def setInteractionPoroNeoHookean(self):
        youngsModulusSI = self.parameters["CELL"]["YoungsModulusSI"]
        volumeFraction = self.parameters["CELL"]["PorousVolumeFraction"]
        youngsModulus = youngsModulusSI / self.parameters["ReynoldScaling"] / self.parameters["CELL"]["YoungsScaling"] / self.p0 
        shearModulus = youngsModulus / (2 * (1+0.5))
        interPNH = inter.InteractionPoroNeoHookean(self.numPoints, self.numTetra, self.force.buf, self.points.buf, self.tetras.buf, self.cell.edgeVectors, self.cell.volumes, shearModulus, volumeFraction, self.vonMises.buf, self.pressure.buf)
        self.interactions.append(interPNH)
        
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
    def setInteractionOverDamped(self, point3fix):
        interOD = inter.InteractionOverDamped(self.numPoints, self.numTetra, self.force.buf, self.points.buf, point3fix=point3fix)
        self.interactions.append(interOD)
        
    def timeStep(self):
        for interaction in self.interactions:
            interaction.beforeTimeStep(self.time)
        for interaction in self.interactions:
            interaction.queueKernel()
        self.time += 1

    def writeVTK(self, writeStress=True, writePressure=True, writeVolumeChange=False):
        # get current coordinates from VRAM and input them into the cell objects mesh
        positions = self.points.readFromGPU()
        positions = positions.reshape((3, self.numPoints)).transpose()
        if np.any(np.isnan(positions)):
            print("\n\n")
            print("+---------------------------------------------------------------------------+")
            print("| !!! Encountered NaN when reading positions from VRAM, exiting program !!! |")
            print("+---------------------------------------------------------------------------+")
            exit()
        self.cell.mesh.points = positions
        if writeStress:
            stress = self.vonMises.readFromGPU()
            self.cell.addTetraData("vonMises", stress)
        if writePressure:
            pressure = -self.pressure.readFromGPU()
            self.cell.addTetraData("pressure", pressure)
        if writeVolumeChange:
            self.cell.calcVolumeChange()
        # write a vtk file named with the current timestep
        self.cell.mesh.save(self.vtkfilespath+f"{self.time}.vtk")

    def run(self, writeStress=True, writePressure=True, writeVolumeChange=False):
        print("\n\n")
        print("+------------------------------------+")
        print("|         Starting Simulation        |")
        print("+------------------------------------+")
        print("\n\n")
        starttimes = []
        endtimes = []
        numberCycles = self.parameters["VTKFramesTotal"]
        dataPoints = None
        for i in range(numberCycles):
            anatime1 = time.time()
            self.writeVTK(writeStress = writeStress, writePressure=writePressure, writeVolumeChange=writeVolumeChange)
            newDataPoints = []
            for quant in self.recordedQuantities:
                newDataPoints.append(quant())
            if self.time == 0: # if first frame
                dataPoints = np.array(newDataPoints)
            else:
                dataPoints = np.vstack((dataPoints, newDataPoints))
            np.savetxt(self.parameters["OutputDir"] + "simdata.dat", dataPoints)
            anatime2 = time.time()
            starttimes.append(time.time())
            for t in range(self.VTKInterval):
                self.timeStep()
            queue.finish()
            endtimes.append(time.time())
            cycleTime = np.average(np.array(endtimes) - np.array(starttimes))
            vtktime = anatime2-anatime1
            print(f"| Average # of simulation steps per second: {np.round(self.VTKInterval / cycleTime, 1)} | VTK cycle took {np.round(vtktime,1)} seconds | estimate of time remaining: {np.round(((numberCycles - i) * (cycleTime + vtktime))/ 60, 1)} min", end="\r")
        self.writeVTK(writeStress = writeStress, writePressure=writePressure, writeVolumeChange=writeVolumeChange)
        print("\n\n")
        print("+------------------------------------+")
        print("|  Simulation finished successfully  |")
        print("+------------------------------------+")
















