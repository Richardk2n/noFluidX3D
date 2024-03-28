#!/bin/sh
echo "gmsh version: " 2>&1 | tee gmsh.log
gmsh -version 2>&1 | tee -a gmsh.log
if [ -z "$1"]; then
  gmsh double_octasphere.geo -3 -smooth 100 -optimize_netgen -algo hxt -format msh2 -o mesh.msh 2>&1 | tee -a gmsh.log
else
  gmsh "$1" -3 -smooth 100 -optimize_netgen -algo hxt -format vtk -o mesh.vtk 2>&1 | tee -a gmsh.log
fi
