import pyvista as pv
import numpy as np

class TetraCell:
    
    def initFromVTK(self, path):
        self.mesh = pv.core.pointset.UnstructuredGrid(pv.read(path), inplace=True)
        self.points = self.mesh.points
        transvec = -self.mesh.points[0]     # point 0 of the reference mesh marks the origin
        self.mesh.translate(transvec)
        self.tetras = self.mesh.cells_dict[10]
        # Calculate the reference Edge vectors and the volumes of each tetrahedra
        self.edgeVectors = []
        self.volumes = []
        for tetra in self.tetras:
            p1, p2, p3, p4 = tetra
            tetraEdgeVectors = np.array([self.points[p1] - self.points[p4], self.points[p2] - self.points[p4], self.points[p3] - self.points[p4]])
            self.edgeVectors.append(tetraEdgeVectors.reshape(9))
            self.volumes.append(1.0/6.0 * abs(np.dot(np.cross(tetraEdgeVectors[0], tetraEdgeVectors[1]), tetraEdgeVectors[2])))
        self.edgeVectors = np.array(self.edgeVectors)
        self.volumes = np.array(self.volumes)
