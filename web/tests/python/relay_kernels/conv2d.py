
import os
from tvm import relay
from tvm.relay import analysis
from tvm.relay.testing.init import create_workload


def convert_conv2d_shape(op_shape_str):
  shapes = op_shape_str.split(",")
  assert len(shapes) == 6 or len(shapes) == 7 or len(shapes) == 11
  if len(shapes) == 6:
    b = 1
    ih = iw = int(shapes[0])
    ic = int(shapes[1])
    kh = kw = int(shapes[2])
    oc = int(shapes[3])
    sh = sw = int(shapes[4])
    ph = pw = int(shapes[5])
  if len(shapes) == 7:
    b = int(shapes[0])
    ih = iw = int(shapes[1])
    ic = int(shapes[2])
    kh = kw = int(shapes[3])
    oc = int(shapes[4])
    sh = sw = int(shapes[5])
    ph = pw = int(shapes[6])
  elif len(shapes) == 11:
    b = int(shapes[0])
    ih = int(shapes[1])
    iw = int(shapes[2])
    ic = int(shapes[3])
    kh = int(shapes[4])
    kw = int(shapes[5])
    oc = int(shapes[6])
    sh = int(shapes[7])
    sw = int(shapes[8])
    ph = int(shapes[9])
    pw = int(shapes[10])
  return b, ih, iw, ic, kh, kw, oc, sh, sw, ph, pw


def build_conv2d_expr(batch_size, ih, iw, ic, kh, kw, oc, sh=1, sw=1, ph=0, pw=0,
                      data_layout="NHWC", kernel_layout="HWIO", dtype="int8"):
  data_shape = (batch_size, ih, iw, ic)
  if data_layout == "NCHW":
    data_shape = (batch_size, ic, ih, iw)
  kernel_shape = (kh, kw, ic, oc)
  if kernel_layout == "OIHW":
    kernel_shape = (oc, ic, kh, kw)
  data = relay.var("data", shape=data_shape, dtype=dtype)
  conv2d_algo = os.getenv("TVM_CONV2D_ALGO")
  assert conv2d_algo is not None
  if conv2d_algo == "conv2d":
    w = relay.var("weight", shape=(kh, kw, ic, oc), dtype=dtype)
    y = relay.nn.conv2d(
        data,
        w,
        strides=(sh, sw),
        padding=(ph, pw),
        channels=oc,
        kernel_size=(kh, kw),
        kernel_layout="HWIO",
        data_layout="NHWC",
        out_dtype=dtype
    )
  elif "conv2d_gemm" in conv2d_algo:
    if "without_weight_transform" in conv2d_algo:
      print("Conv2D without weight transform")
      #tile_rows_B = 32  # For k
      #tile_cols_B = 8  # For n
      tile_rows_B = os.getenv("TVM_GEMM_KB")
      tile_cols_B = os.getenv("TVM_GEMM_NB")
      tile_rows_B = 32 if tile_rows_B is None else int(tile_rows_B)
      tile_cols_B = 8 if tile_cols_B is None else int(tile_cols_B)
      k = kh * kw * ic
      n = oc
      ko, ki = k // tile_rows_B, tile_rows_B
      no, ni = n // tile_cols_B, tile_cols_B
      w = relay.var("weight", shape=(ko, no, ki, ni), dtype=dtype)
      y = relay.nn.contrib_conv2d_gemm_without_weight_transform(
          data,
          w,
          strides=(sh, sw),
          padding=(ph, pw),
          channels=oc,
          kernel_size=(kh, kw),
          kernel_layout="HWIO",
          data_layout="NHWC",
          out_dtype=dtype
      )
    else:
      w = relay.var("weight", shape=(kh, kw, ic, oc), dtype=dtype)
      y = relay.nn.contrib_conv2d_gemm(
          data,
          w,
          strides=(sh, sw),
          padding=(ph, pw),
          channels=oc,
          kernel_size=(kh, kw),
          kernel_layout="HWIO",
          data_layout="NHWC",
          out_dtype=dtype
      )
  args = analysis.free_vars(y)
  workload = relay.Function(args, y)

  return workload, [data, w, y]


def build_conv2d_workload(op_shape_str, data_layout="NHWC", kernel_layout="HWIO", dtype="int8"):
  #in_size = 56; in_ch = 256; k_size = 3; out_ch = 512
  #in_size = 12; in_ch = 384; k_size = 3; out_ch = 384

  b, ih, iw, ic, kh, kw, oc, sh, sw, ph, pw = convert_conv2d_shape(op_shape_str)
  
  expr, mod_args = build_conv2d_expr(b, ih, iw, ic, kh, kw, oc, sh, sw, ph, pw, data_layout, kernel_layout, dtype)
  mod, params = create_workload(expr)
  return mod, params, "conv2d"
