import numpy as np
import tvm
from tvm import te


# The sizes of inputs and filters
batch = 256
in_channel = 256
out_channel = 256  # 512
in_size = 14  # 14
kernel = 3
pad = 1
stride = 1

# Algorithm
A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
W = te.placeholder((kernel, kernel, in_channel, out_channel), name="W")
out_size = (in_size - kernel + 2 * pad) // stride + 1
# Pad input
Apad = te.compute(
    (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.tir.if_then_else(
        tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn],
        tvm.tir.const(0.0, "float32"),
    ),
    name="Apad",
)
# Create reduction variables
rc = te.reduce_axis((0, in_channel), name="rc")
ry = te.reduce_axis((0, kernel), name="ry")
rx = te.reduce_axis((0, kernel), name="rx")
# Compute the convolution
B = te.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: te.sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
    ),
    name="B",
)

# Designate the memory hierarchy
s = te.create_schedule(B.op)
s[Apad].compute_inline()  # compute Apad inline
AA = s.cache_read(Apad, "shared", [B])
WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(AA, "local", [B])
WL = s.cache_read(WW, "local", [B])
BL = s.cache_write(B, "local")

# tile consts
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# Get the GPU thread indices
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

# Split the workloads
hi, wi, fi, ni = s[B].op.axis
bz = s[B].fuse(hi, wi)
by, fi = s[B].split(fi, factor=block_factor)
bx, ni = s[B].split(ni, factor=block_factor)

# Bind the iteration variables to GPU thread indices
s[B].bind(bz, block_z)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)

tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
ty, fi = s[B].split(fi, nparts=num_thread)
tx, ni = s[B].split(ni, nparts=num_thread)
s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

s[B].bind(tyz, thread_yz)
s[B].bind(txz, thread_xz)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)

# Schedule BL local write
s[BL].compute_at(s[B], tx)
yi, xi, fi, ni = s[BL].op.axis
ry, rx, rc = s[BL].op.reduce_axis
rco, rci = s[BL].split(rc, factor=step)
s[BL].reorder(rco, ry, rx, rci, fi, ni)

# Attach computation to iteration variables
s[AA].compute_at(s[BL], rx)
s[WW].compute_at(s[BL], rx)
s[AL].compute_at(s[BL], rci)
s[WL].compute_at(s[BL], rci)

# Schedule for A's shared memory load
yi, xi, ci, ni = s[AA].op.axis
ty, ci = s[AA].split(ci, nparts=num_thread)
tx, ni = s[AA].split(ni, nparts=num_thread)
_, ni = s[AA].split(ni, factor=4)
s[AA].reorder(ty, tx, yi, xi, ci, ni)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)
s[AA].vectorize(ni)  # vectorize memory load

# Schedule for W's shared memory load
yi, xi, ci, fi = s[WW].op.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
_, fi = s[WW].split(fi, factor=4)
s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)
s[WW].vectorize(fi)  # vectorize memory load

#target = "cuda"
target = tvm.target.cuda(model="nvidia/geforce-rtx-3050", arch="sm_86")
func = tvm.build(s, [A, W, B], target)
dev = tvm.cuda(0)
a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype(A.dtype)
w_np = np.random.uniform(size=(kernel, kernel, in_channel, out_channel)).astype(W.dtype)
a = tvm.nd.array(a_np, dev)
w = tvm.nd.array(w_np, dev)
b = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=B.dtype), dev)
func(a, w, b)
evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Convolution: %f ms" % (evaluator(a, w, b).mean * 1e3))
