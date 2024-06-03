
import onnx
import tvm.relay as relay


def roberta_model(dtype="float32"):
  batch_size = 1
  seq_len = 384

  if dtype == "int8":
    onnx_model = onnx.load("onnx_models/roberta-quant.onnx")
  else:
    onnx_model = onnx.load("onnx_models/roberta.onnx")
    shape_dict = {"input_ids": (batch_size, seq_len),
                  "attention_mask": (batch_size, seq_len)}
    #onnx_model = onnx.load("onnx_models/roberta-small.onnx")
    #shape_dict = {"input": (batch_size, seq_len)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
