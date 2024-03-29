
import os
import argparse
from multiprocessing import Process
from kernel_tuning_common import OpType, TunerType, string_to_op_type, string_to_tuner_type
from backend_tools import string_to_backend
from kernel_tuning_test import build_and_tune_kernel


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--op-type", type=str, required=True, help="Op type")
  parser.add_argument("--op-shape", type=str, required=True, help="Op shape")
  parser.add_argument("--data-type", type=str, choices=["float32", "int8"],
                      default="float32", help="Data type")
  parser.add_argument("--backend", type=str,
                      choices=["cuda", "llvm-wasm", "wasm", "webgpu"],
                      default="wasm", help="Backend")
  parser.add_argument("--dev-info", type=str, default="unknown",
                      help="Device infomation")
  parser.add_argument("--tune", action="store_true", help="Enable tuning")
  parser.add_argument("--number", type=int, default=50,
                      help="Number of rounds")

  return parser.parse_args()


def fast_tune_common_kernel(
    op_type,
    op_shape,
    dtype,
    dev_info,
    backend,
    enable_tuning,
    number
):
  build_and_tune_kernel(
    op_type,
    op_shape,
    dtype,
    tunner_type=TunerType.Fast,
    dev_info=dev_info,
    target_backend=backend,
    enable_tune=enable_tuning,
    number=number
  )


def fast_tune_packed_matmul_kernel(
    op_type,
    op_shape,
    dtype,
    dev_info,
    backend,
    enable_tuning,
    number
):
  shapes = op_shape.split(",")
  assert len(shapes) == 4
  m = int(shapes[1])
  k = int(shapes[2])
  n = int(shapes[3])
  
  mb_candidates = [4, 8, 16, 32, 64, 128, 256]
  kb_candidates = [4, 8, 16, 32, 64, 128, 256]
  nb_candidates = [4, 8, 16, 32, 64, 128, 256]
  #mb_candidates = [256]
  #kb_candidates = [16]
  #nb_candidates = [256]

  def set_size_in_envs(m, k, n, mb, kb, nb):
    os.environ["TVM_GEMM_M"] = str(m)
    os.environ["TVM_GEMM_K"] = str(k)
    os.environ["TVM_GEMM_N"] = str(n)
    os.environ["TVM_GEMM_MB"] = str(mb)
    os.environ["TVM_GEMM_KB"] = str(kb)
    os.environ["TVM_GEMM_NB"] = str(nb)

  for kb in kb_candidates:
    for mb in mb_candidates:
      for nb in nb_candidates:
        if mb != nb:
          continue
        set_size_in_envs(m, k, n, mb, kb, nb)
        final_op_shape = op_shape + ",%d,%d,%d" % (mb, kb, nb)
        try:
          build_and_tune_kernel(
            op_type,
            final_op_shape,
            dtype,
            tunner_type=TunerType.Fast,
            dev_info=dev_info,
            target_backend=backend,
            enable_tune=enable_tuning,
            number=number
          )
        except ValueError:
          print("Error: Build and tune kernel failed")


def fast_tune_kernel(
    op_type,
    op_shape,
    dtype,
    dev_info,
    backend,
    enable_tuning,
    number
):
  if op_type == OpType.Dense:
    os.environ["TVM_KERNEL_NAME"] = "dense"
    fast_tune_common_kernel(op_type, op_shape, dtype, dev_info, backend,
                            enable_tuning, number)
  elif op_type in [OpType.MatMul, OpType.BatchMatMul]:
    os.environ["TVM_KERNEL_NAME"] = "matmul"
    fast_tune_common_kernel(op_type, op_shape, dtype, dev_info, backend,
                            enable_tuning, number)
  elif op_type == OpType.PackedBatchMatMul:
    os.environ["TVM_KERNEL_NAME"] = "matmul"
    fast_tune_packed_matmul_kernel(
      op_type,
      op_shape,
      dtype,
      dev_info,
      backend,
      enable_tuning,
      number
    )
  elif op_type == OpType.Conv2d:
    os.environ["TVM_KERNEL_NAME"] = "conv2d"
    fast_tune_common_kernel(op_type, op_shape, dtype, dev_info, backend,
                            enable_tuning, number)
  else:
    raise ValueError("Unsupported op type: " + str(op_type))


if __name__ == "__main__":
  args = parse_args()
  fast_tune_kernel(
    string_to_op_type(args.op_type),
    args.op_shape,
    args.data_type,
    args.dev_info,
    string_to_backend(args.backend),
    args.tune,
    args.number
  )
