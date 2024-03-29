
from tvm import relay
from tvm.relay import analysis
from tvm.relay.testing.init import create_workload


def convert_dense_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 3
  return int(shapes[0]), int(shapes[1]), int(shapes[2])


def build_dense_expr(batch, units, units_in, dtype):
  data = relay.var("data", shape=(batch, units_in), dtype=dtype)
  w = relay.var("weight", shape=(units, units_in), dtype=dtype)
  y = relay.nn.dense(
      data,
      w,
      units=units,
      out_dtype=dtype
  )
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, w, y]


def build_dense_workload(op_shape_str, dtype="int8"):
  batch, units, units_in = convert_dense_shape(op_shape_str)
  expr, mod_args = build_dense_expr(batch, units, units_in, dtype)
  mod, params = create_workload(expr)
  return mod, params, "dense"


def convert_packed_dense_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 3
  return int(shapes[0]), int(shapes[1]), int(shapes[2])


def build_packed_dense_expr(batch, units, units_in, dtype):
  data = relay.var("data", shape=(batch, units_in), dtype=dtype)
  w = relay.var("weight", shape=(units, units_in), dtype=dtype)
  y = relay.nn.contrib_dense_pack(
      data,
      w,
      units=units,
      out_dtype=dtype
  )
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, w, y]


def build_packed_dense_workload(op_shape_str, dtype="int8"):
  batch, units, units_in = convert_packed_dense_shape(op_shape_str)
  expr, mod_args = build_packed_dense_expr(batch, units, units_in, dtype)
  mod, params = create_workload(expr)
  return mod, params, "packed-dense"
