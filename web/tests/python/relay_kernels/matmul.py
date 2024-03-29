
from tvm import relay
from tvm.relay import analysis
from tvm.relay.testing.init import create_workload


def convert_matmul_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 3
  return int(shapes[0]), int(shapes[1]), int(shapes[2])


def build_matmul_expr(m, k, n, dtype, transpose_a=False, transpose_b=True):
  shape_data = (m, k) if not transpose_a else (k, m)
  data = relay.var("data", shape=shape_data, dtype=dtype)
  shape_w = (k, n) if not transpose_b else (n, k)
  w = relay.var("weight", shape=shape_w, dtype=dtype)
  y = relay.nn.matmul(
      data,
      w,
      out_dtype=dtype,
      transpose_a=transpose_a,
      transpose_b=transpose_b
  )
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, w, y]


def build_matmul_workload(op_shape_str, dtype="int8"):
  #m = 512; k = 512; n = 512
  m, k, n = convert_matmul_shape(op_shape_str)
  expr, mod_args = build_matmul_expr(m, k, n, dtype)
  mod, params = create_workload(expr)
  return mod, params, "matmul"
