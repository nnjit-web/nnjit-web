
import tvm.relay as relay


def mlp_model(batch_size=1, layout="NCHW", dtype="float32"):
  # auto-scheduler prefers NHWC layout
  if layout == "NHWC":
      image_shape = (224, 224, 3)
  elif layout == "NCHW":
      image_shape = (3, 224, 224)
  else:
      raise ValueError("Invalid layout: " + layout)
  
  mod, params = relay.testing.mlp.get_workload(
      batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
  )
  return mod, params
