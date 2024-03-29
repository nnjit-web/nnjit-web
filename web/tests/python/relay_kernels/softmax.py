
from tvm import relay
from tvm.relay import analysis
from tvm.relay.testing.init import create_workload


def convert_softmax_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 2
  return int(shapes[0]), int(shapes[1])


def build_softmax_expr(n, c, dtype):
  data = relay.var("data", shape=(n, c), dtype=dtype)
  y = relay.nn.softmax(data)
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, y]


def build_softmax_workload(op_shape_str, dtype="int8"):
  n, c = convert_softmax_shape(op_shape_str)
  expr, mod_args = build_softmax_expr(n, c, dtype)
  mod, params = create_workload(expr)
  return mod, params, "softmax"
