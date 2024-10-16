# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Wed Oct 16 12:29:40 2024

@author: Richard Kellnberger
"""

import pyopencl as cl

device: cl.Device = None
context: cl.Context = None
commandQueue: cl.CommandQueue = None


def initializeOpenCLObjects(deviceId: int):
    global device, context, commandQueue
    runningId: int = 0
    found: bool = False
    for p in cl.get_platforms():
        devices = p.get_devices()
        numberDevices = len(devices)
        if runningId + numberDevices <= deviceId:
            runningId += numberDevices
        else:
            device = devices[deviceId - runningId]
            found = True
            break
    if not found:
        print(f"ERROR: Device ID {deviceId} does not exist!\n")
        exit(1)
    context = cl.Context([device])
    commandQueue = cl.CommandQueue(context)


# TODO unused?
def getDevice() -> cl.Device:
    return device


def getContext() -> cl.Context:
    return context


def getCommandQueue() -> cl.CommandQueue:
    return commandQueue


__all__ = ["initializeOpenCLObjects", "getDevice", "getContext", "getCommandQueue"]
