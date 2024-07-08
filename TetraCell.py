import pyvista as pv
import numpy as np

class TetraCell:
    
    def initFromVTK(self, path, radius=18.0, recenter=True):
        self.mesh = pv.core.pointset.UnstructuredGrid(pv.read(path), inplace=True)
        if recenter:
            transvec = -self.mesh.points[0]     # point 0 of the reference mesh marks the origin
            self.mesh.translate(transvec)
            #radiusMesh = np.linalg.norm(self.mesh.points[1])
            radiusMesh = np.abs(self.mesh.points[1][0]) # x-value of point1
            returnScale = self.mesh.scale(radius / radiusMesh)
            if returnScale != None:
                self.mesh = returnScale
        self.points = self.mesh.points
        self.radius = radius
        self.tetras = self.mesh.cells_dict[10]
        # Calculate the reference Edge vectors and the volumes of each tetrahedra
        p1 = self.points[self.tetras[:, 0]]
        p2 = self.points[self.tetras[:, 1]]
        p3 = self.points[self.tetras[:, 2]]
        p4 = self.points[self.tetras[:, 3]]

        edgeVectors = np.array([p1-p4, p2-p4, p3-p4])
        edgeVectors = np.einsum("ijk -> jik", edgeVectors)
        self.edgeVectors = edgeVectors.reshape(len(edgeVectors), 9)
        self.volumes = 1.0/6.0 * np.abs( np.linalg.det(edgeVectors))
      
    def scale(self, scaling):
        returnScale = self.mesh.scale(scaling)
        if returnScale != None:
            self.mesh = returnScale
        self.points = self.mesh.points
        self.edgeVectors = []
        self.volumes = []
        p1 = self.points[self.tetras[:, 0]]
        p2 = self.points[self.tetras[:, 1]]
        p3 = self.points[self.tetras[:, 2]]
        p4 = self.points[self.tetras[:, 3]]

        edgeVectors = np.array([p1-p4, p2-p4, p3-p4])
        edgeVectors = np.einsum("ijk -> jik", edgeVectors)
        self.edgeVectors = edgeVectors.reshape(len(edgeVectors), 9)
        self.volumes = 1.0/6.0 * np.abs( np.linalg.det(edgeVectors))
      
    def addTetraData(self, name, dataArray):
        # pad data with zeros
        dataArrayPadded = np.pad(dataArray, pad_width=(self.mesh.n_cells - dataArray.size, 0))
        self.mesh.cell_data.set_array(dataArrayPadded, name)
            
    def calcVolumeChange(self):
        curVolumes = []
        for tetra in self.tetras:
            p1, p2, p3, p4 = tetra
            tetraEdgeVectors = np.array([self.mesh.points[p1] - self.mesh.points[p4], self.mesh.points[p2] - self.mesh.points[p4], self.mesh.points[p3] - self.mesh.points[p4]])
            curVolumes.append(1.0/6.0 * abs(np.dot(np.cross(tetraEdgeVectors[0], tetraEdgeVectors[1]), tetraEdgeVectors[2])))
        curVolumes = np.array(curVolumes)
        self.addTetraData("volumeChange", curVolumes/self.volumes - 1)

    def calcAverageCellDataValue(self, dataname):
        sumData = self.mesh.cell_data[dataname].sum()
        return sumData / self.tetras[:, 0].size
class TetraCellDeformed(TetraCell):
    def initFromVTK(self, path, radius=18.0):
        self.mesh = pv.core.pointset.UnstructuredGrid(pv.read(path), inplace=True)
        self.points = self.mesh.points
        self.tetras = self.mesh.cells_dict[10]
        self.edgeVectors = []
        self.volumes = []
        for tetra in self.tetras:
            p1, p2, p3, p4 = tetra
            tetraEdgeVectors = np.array([self.points[p1] - self.points[p4], self.points[p2] - self.points[p4], self.points[p3] - self.points[p4]])
            self.edgeVectors.append(tetraEdgeVectors.reshape(9))
            self.volumes.append(1.0/6.0 * abs(np.dot(np.cross(tetraEdgeVectors[0], tetraEdgeVectors[1]), tetraEdgeVectors[2])))
        self.edgeVectors = np.array(self.edgeVectors)
        self.volumes = np.array(self.volumes)
