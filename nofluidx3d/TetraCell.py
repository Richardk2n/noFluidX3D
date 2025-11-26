import numpy as np
import pyvista as pv
from numpy.linalg import vecdot

try:
    from warnings import deprecated  # Python 3.13
except ImportError:
    from nofluidx3d.util.decorators import deprecated


class TetraCell:

    def __init__(self, path, radius=18.0, recenter=True):
        self.mesh = pv.core.pointset.UnstructuredGrid(pv.read(path), inplace=True)
        if recenter:
            transvec = -self.mesh.points[0]  # point 0 of the reference mesh marks the origin
            self.mesh.translate(transvec)
            # radiusMesh = np.linalg.norm(self.mesh.points[1])
            radiusMesh = np.abs(self.mesh.points[1][0])  # x-value of point1
            self.mesh.scale(radius / radiusMesh, inplace=True)
        self.points = self.mesh.points
        self.radius = radius
        self.tetras = self.mesh.cells_dict[10]
        # Calculate the reference Edge vectors and the volumes of each tetrahedra

        R1, R2, R3 = self.calcEdgeVectors(self.points)

        edgeVectors = np.array([R1, R2, R3])
        edgeVectors = np.einsum("ijk -> jik", edgeVectors)
        self.edgeVectors = edgeVectors.reshape(len(edgeVectors), 9)
        self.volumes = self.calcVolumes(R1, R2, R3)

    def calcEdgeVectors(self, points):
        p0 = points[self.tetras[:, 0]]
        p1 = points[self.tetras[:, 1]]
        p2 = points[self.tetras[:, 2]]
        p3 = points[self.tetras[:, 3]]
        return p1 - p0, p2 - p0, p3 - p0

    def calcVolumes(self, R1, R2, R3):
        return vecdot(R1, np.cross(R2, R3)) / 6

    def scale(self, scaling):
        self.mesh.scale(scaling, inplace=True)
        self.points = self.mesh.points
        R1, R2, R3 = self.calcEdgeVectors(self.points)

        edgeVectors = np.array([R1, R2, R3])
        edgeVectors = np.einsum("ijk -> jik", edgeVectors)
        self.edgeVectors = edgeVectors.reshape(len(edgeVectors), 9)
        self.volumes = self.calcVolumes(R1, R2, R3)

    def addTetraData(self, name, dataArray):
        # pad data with zeros
        dataArrayPadded = np.pad(dataArray, pad_width=(self.mesh.n_cells - dataArray.size, 0))
        self.mesh.cell_data.set_array(dataArrayPadded, name)

    def calcVolumeChange(self):
        curVolumes = self.calcVolumes(*self.calcEdgeVectors(self.mesh.points))
        self.addTetraData("volumeChange", curVolumes / self.volumes - 1)

    @deprecated("No idea what this is")
    def calcAverageCellDataValue(self, dataname):
        sumData = self.mesh.cell_data[dataname].sum()
        return sumData / self.tetras[:, 0].size


@deprecated("Use TetraCell instead")
class TetraCellDeformed(TetraCell):
    @deprecated("Use TetraCell instead")
    def __init__(self, path, radius=18.0):
        self.mesh = pv.core.pointset.UnstructuredGrid(pv.read(path), inplace=True)
        self.points = self.mesh.points
        self.tetras = self.mesh.cells_dict[10]
        self.edgeVectors = []
        self.volumes = []
        for tetra in self.tetras:
            p0, p1, p2, p3 = tetra
            tetraEdgeVectors = np.array(
                [
                    self.points[p1] - self.points[p0],
                    self.points[p2] - self.points[p0],
                    self.points[p3] - self.points[p0],
                ]
            )
            self.edgeVectors.append(tetraEdgeVectors.reshape(9))
            self.volumes.append(
                1.0
                / 6.0
                * abs(
                    np.dot(np.cross(tetraEdgeVectors[0], tetraEdgeVectors[1]), tetraEdgeVectors[2])
                )
            )
        self.edgeVectors = np.array(self.edgeVectors)
        self.volumes = np.array(self.volumes)
