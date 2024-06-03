
import onnx
import tvm.relay as relay


def t5_encoder_model():
  onnx_model = onnx.load("onnx_models/t5-small-encoder.onnx")
  shape_dict = {"input_ids": (1, 256),
                "attention_mask": (1, 256)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params


def t5_decoder_model():
  onnx_model = onnx.load("onnx_models/t5-small-decoder.onnx")
  shape_dict = {"input_ids": (1, 256),
                "encoder_hidden_states": (1, 256, 512)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
