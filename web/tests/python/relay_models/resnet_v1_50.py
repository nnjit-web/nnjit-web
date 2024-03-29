
import onnx
import numpy as np
from PIL import Image
import tvm.relay as relay
from tvm.contrib.download import download_testdata


def preprocess_img_data(img_path):
  # Resize it to 224x224
  resized_image = Image.open(img_path).resize((224, 224))
  img_data = np.asarray(resized_image).astype("float32")

  # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
  img_data = np.transpose(img_data, (2, 0, 1))

  # Normalize according to the ImageNet input specification
  imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
  imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
  norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

  # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
  img_data = np.expand_dims(norm_img_data, axis=0)

  return img_data


def resnet_v1_50_model():
  # Download model.
  model_url = (
      "https://github.com/onnx/models/raw/main/"
      "vision/classification/resnet/model/"
      "resnet50-v1-7.onnx"
  )

  model_path = download_testdata(model_url, "resnet50-v1-7.onnx", module="onnx")
  onnx_model = onnx.load(model_path)

  # Download image.
  img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
  img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

  # Preprocess image data.
  img_data = preprocess_img_data(img_path)

  # The input name may vary across model types. You can use a tool
  # like Netron to check input names
  input_name = "data"
  shape_dict = {input_name: img_data.shape}

  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

  return mod, params
