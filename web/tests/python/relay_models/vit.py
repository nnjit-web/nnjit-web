
import onnx
import tvm.relay as relay


def vit_model():
  #onnx_model = onnx.load("onnx_models/vit.onnx")
  #shape_dict = {"pixel_values": (1, 3, 224, 224)}
  
  onnx_model = onnx.load("onnx_models/vit-s384.onnx")
  shape_dict = {"input": (1, 3, 384, 384)}

  #onnx_model = onnx.load("onnx_models/deit-s.onnx")
  #shape_dict = {"input": (1, 3, 384, 384)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
