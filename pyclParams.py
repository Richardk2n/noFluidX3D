import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
