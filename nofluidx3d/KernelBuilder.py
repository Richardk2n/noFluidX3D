# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Tue Mar 25 15:18:15 2025

@author: Richard Kellnberger
"""
from pathlib import Path
from typing import Any

import pyopencl as cl

from nofluidx3d.openCL import getContext

kernels = Path(__file__).parent / "kernels"
dependencies = [
    kernels / "typedefs.h",
    kernels / "atomicAdd.cl",
    kernels / "matrix.h",
    kernels / "matrix.cl",
]


class KernelBuilder:

    defines: set[str] = set()
    valuesDefines: dict[str, Any] = {}

    @staticmethod
    def readDependencies() -> str:
        deps = ""
        for d in dependencies:
            with open(d, "r") as f:
                deps += f.read() + "\n"
        return deps

    @staticmethod
    def define(*args, **kwargs):
        KernelBuilder.defines.update(set(args))
        KernelBuilder.valuesDefines.update(kwargs)

    @staticmethod
    def getDefines() -> str:
        defines = ""
        for d in KernelBuilder.defines:
            defines += f"#define {d}\n"
        for name, value in KernelBuilder.valuesDefines.items():
            defines += f"#define {name} {value}\n"
        return defines

    @staticmethod
    def build(file: Path, kernelName: str):
        with open(file, "r") as f:
            kernelStr = f.read() + "\n"

        string = KernelBuilder.getDefines() + KernelBuilder.readDependencies() + kernelStr

        prg = cl.Program(getContext(), string).build()
        return prg.__getattr__(kernelName)
