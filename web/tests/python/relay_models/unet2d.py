
import onnx
import tvm.relay as relay


def unet2d_model():
  onnx_model = onnx.load("onnx_models/unet2d-s32.onnx")
  shape_dict = {"image": (1, 3, 32, 32),
                "timestep": (1)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
