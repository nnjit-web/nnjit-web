
import onnx
import tvm.relay as relay


def mobilebert_model():
  model_filename = "mobilebert-s384.onnx"
  onnx_model = onnx.load("onnx_models/" + model_filename)

  batch_size = 1
  seq_len = 384
  if model_filename == "mobilebert.onnx":
    shape_dict = {"input_ids": (batch_size, seq_len),
                  "attention_mask": (batch_size, seq_len),
                  "token_type_ids": (batch_size, seq_len)}
  elif model_filename == "mobilebert-s384.onnx":
    shape_dict = {"input": (batch_size, seq_len)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
