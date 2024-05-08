import TetraCell as tc
import numpy as np
import matplotlib.pyplot as plt

vtk = "/tp6-gekle/nas/bt709186/noFluidx3d_results/microgel_tip_NH/vtkfiles/frame18625404.vtk"

cell = tc.TetraCellDeformed()
cell.initFromVTK(vtk)

vonmises = cell.mesh.cell_data.get("vonMises")[-cell.tetras[:,0].size:]
