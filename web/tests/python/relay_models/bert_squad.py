
import onnx
import tvm.relay as relay


def bert_squad_model():
  onnx_model = onnx.load("/mnt/d/Downloads/onnx_models/bertsquad-12.onnx")

  seq_len = 64
  shape_dict = {"unique_ids_raw_output___9:0": (seq_len),
                "segment_ids:0": (seq_len, 256),
                "input_mask:0": (seq_len, 256),
                "input_ids:0": (seq_len, 256)}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
