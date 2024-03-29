
from enum import Enum


class TargetBackend(Enum):
  CUDA = 0
  LLVM_WASM = 1
  WASM = 2
  TVMWebGPU = 3
  WebGPU = 4


def string_to_backend(backend_str):
  backend_dict = {"cuda": TargetBackend.CUDA,
                  "llvm-wasm": TargetBackend.LLVM_WASM,
                  "wasm": TargetBackend.WASM,
                  "tvm-webgpu": TargetBackend.TVMWebGPU,
                  "webgpu": TargetBackend.WebGPU}
  if backend_str not in backend_dict:
    raise ValueError("Unsupported backend " + backend_str)
  return backend_dict[backend_str]


def backend_to_string(backend):
  backend_dict = {TargetBackend.CUDA: "cuda",
                  TargetBackend.LLVM_WASM: "llvm-wasm",
                  TargetBackend.WASM: "wasm",
                  TargetBackend.TVMWebGPU: "tvm-webgpu",
                  TargetBackend.WebGPU: "webgpu"}
  if backend not in backend_dict:
    raise ValueError("Unsupported backend " + str(backend))
  return backend_dict[backend]


def is_wasm_backend(backend):
  if isinstance(backend, TargetBackend):
    return backend == TargetBackend.LLVM_WASM or backend == TargetBackend.WASM
  if isinstance(backend, str):
    return backend == "llvm-wasm" or backend == "wasm"
  return False


def is_webgpu_backend(backend):
  if isinstance(backend, TargetBackend):
    return backend == TargetBackend.TVMWebGPU or backend == TargetBackend.WebGPU
  if isinstance(backend, str):
    return backend == "tvm-webgpu" or backend == "webgpu"
  return False
