
import onnx
import tvm.relay as relay


def bart_model():
  #onnx_model = onnx.load("onnx_models/bart.onnx")
  onnx_model = onnx.load("onnx_models/bart-small.onnx")
  #onnx_model = onnx.load("onnx_models/bart-s384.onnx")

  batch_size = 1
  seq_len = 384
  #shape_dict = {"input_ids": (batch_size, seq_len),
  #              "attention_mask": (batch_size, seq_len)}
  shape_dict = {"input": (batch_size, seq_len)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
