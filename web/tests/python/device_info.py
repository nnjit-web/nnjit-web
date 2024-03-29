
from backend_tools import TargetBackend, is_wasm_backend


def get_processor_info(dev_info, backend):
  processor_info = "unknown"
  if dev_info == "hp-elitedesk-800-g6":
    if is_wasm_backend(backend):
      processor_info = "intel-i7-10700"
    elif backend == TargetBackend.WebGPU:
      processor_info = "intel-hd-graphics-630"
  elif dev_info == "honor-magicbook-16":
    if is_wasm_backend(backend):
      processor_info = "amd-ryzen-7-5800h"
    elif backend == TargetBackend.WebGPU:
      processor_info = "nvidia-rtx-3050-laptop"
  elif dev_info == "pixel-4-xl":
    if is_wasm_backend(backend):
      processor_info = "kryo-485"
    elif backend == TargetBackend.WebGPU:
      processor_info = "adreno-640"
  elif dev_info == "vivo-x30":
    if is_wasm_backend(backend):
      processor_info = "cortex-a77"
    elif backend == TargetBackend.WebGPU:
      processor_info = "mali-g76-mp5"
  elif dev_info == "honor-70":
    if is_wasm_backend(backend):
      processor_info = "kyro-670"
    elif backend == TargetBackend.WebGPU:
      processor_info = "adreno-642l"
  elif dev_info == "honor-9":
    if is_wasm_backend(backend):
      processor_info = "arm-cotex-a73"
    elif backend == TargetBackend.WebGPU:
      processor_info = "mali-g71-mp8"
  elif dev_info == "mate-20":
    if is_wasm_backend(backend):
      processor_info = "cortex-a76"
    elif backend == TargetBackend.WebGPU:
      processor_info = "mali-g76-mp10"

  return processor_info
