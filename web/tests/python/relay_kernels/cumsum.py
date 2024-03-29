
from tvm import relay
from tvm.relay import analysis
from tvm.relay.testing.init import create_workload


def convert_cumsum_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 2
  return int(shapes[0]), int(shapes[1])


def build_cumsum_expr(batch, seq_len, dtype):
  data = relay.var("data", shape=(batch, seq_len), dtype=dtype)
  y = relay.op.cumsum(data, 1, dtype)
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, y]


def build_cumsum_workload(op_shape_str, dtype="float32"):
  batch, seq_len = convert_cumsum_shape(op_shape_str)
  expr, mod_args = build_cumsum_expr(batch, seq_len, dtype)
  mod, params = create_workload(expr)
  return mod, params, "cumsum"
