
import onnx
import tvm.relay as relay


def gpt2_model():
  onnx_model = onnx.load("onnx_models/gpt2-10.onnx")
  input_name = "input1"
  #onnx_model = onnx.load("onnx_models/gpt2-s10x64.onnx")
  #input_name = "input"

  batch_size = 1
  seq_len = 10
  num_tokens = 64
  shape_dict = {input_name: (batch_size, seq_len, num_tokens)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
