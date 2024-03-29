
import os
import tvm.relay as relay


def vgg16_model(batch_size=1, layout="NCHW", dtype="float32"):
  # Set layout for conv gemm
  os.environ["TVM_USE_CONV_GEMM"] = "1"
  layout = "NHWC"

  # auto-scheduler prefers NHWC layout
  if layout == "NHWC":
      image_shape = (224, 224, 3)
  elif layout == "NCHW":
      image_shape = (3, 224, 224)
  else:
      raise ValueError("Invalid layout: " + layout)
  
  mod, params = relay.testing.vgg.get_workload(
      batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000, num_layers=16
  )
  return mod, params
