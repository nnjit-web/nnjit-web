
import os
from tvm import relay
from tvm.relay import analysis
from tvm.relay.testing.init import create_workload


def convert_batch_matmul_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 4
  return int(shapes[0]), int(shapes[1]), int(shapes[2]), int(shapes[3])


def build_batch_matmul_expr(batch_size, m, k, n, transpose_a, transpose_b, dtype):
  shape_a = (batch_size, m, k) if not transpose_a else (batch_size, k, m)
  shape_b = (batch_size, k, n) if not transpose_b else (batch_size, n, k)
  data = relay.var("data", shape=shape_a, dtype=dtype)
  w = relay.var("weight", shape=shape_b, dtype=dtype)
  y = relay.nn.batch_matmul(
      data,
      w,
      out_dtype=dtype,
      transpose_a=transpose_a,
      transpose_b=transpose_b
  )
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, w, y]


def build_batch_matmul_workload(op_shape_str, dtype="int8"):
  #batch_size = 1; m = 480; k = 480; n = 480
  #transpose_a = False; transpose_b = True
  #batch_size = 1; m = 384; k = 512; n = 2 * 128
  #transpose_a = False; transpose_b = True
  #batch_size = 4; m = 10 * 48; k = 10 * 48; n = 10 * 48
  #transpose_a = False; transpose_b = True
  #batch_size = 1; m = 8 * 48; k = 8 * 48; n = 8 * 48
  #transpose_a = False; transpose_b = True
  #batch_size = 4; m = 480; k = 1 * 48; n = 480
  #transpose_a = False; transpose_b = True
  #batch_size = 4; m = 384; k = 1 * 48; n = 384
  #transpose_a = False; transpose_b = True
  #batch_size = 4; m = 384; k = 1 * 32; n = 384
  #transpose_a = False; transpose_b = True

  batch_size, m, k, n = convert_batch_matmul_shape(op_shape_str)
  transpose_a = False; transpose_b = False

  expr, mod_args = build_batch_matmul_expr(batch_size, m, k, n, transpose_a, transpose_b, dtype)
  mod, params = create_workload(expr)
  return mod, params, "batch-matmul"


def convert_packed_batch_matmul_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 7
  return int(shapes[0]), int(shapes[1]), int(shapes[2]), int(shapes[3]), int(shapes[4]), int(shapes[5]), int(shapes[6])


def build_packed_batch_matmul_expr(batch, m, k, n, mb, kb, nb, dtype):
  os.environ["TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM"] = "1"
  data = relay.var("data", shape=(batch, 1, m, k), dtype=dtype)
  w = relay.var("weight", shape=(1, 1, k, n), dtype=dtype)
  y = relay.nn.contrib_conv2d_gemm(
      data,
      w,
      strides=(1, 1),
      padding=(0, 0),
      channels=n,
      kernel_size=(1, 1),
      kernel_layout="HWIO",
      data_layout="NHWC",
      out_dtype=dtype
  )
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, w, y]


def build_packed_batch_matmul_workload(op_shape_str, dtype="int8"):
  batch, m, k, n, mb, kb, nb = convert_packed_batch_matmul_shape(op_shape_str)
  expr, mod_args = build_packed_batch_matmul_expr(batch, m, k, n, mb, kb, nb, dtype)
  mod, params = create_workload(expr)
  return mod, params, "packed-batch-matmul"
