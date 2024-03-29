
from enum import Enum


class OpType(Enum):
  MatMul = 0
  BatchMatMul = 1
  PackedBatchMatMul = 2
  Dense = 3
  PackedDense = 4
  Conv2d = 5
  CumSum = 6
  Softmax = 7


class TunerType(Enum):
  AutoTVM = 0
  AutoScheduler = 1
  Fast = 2


def string_to_op_type(op_str):
  op_type_dict = {"matmul": OpType.MatMul,
                  "batch_matmul": OpType.BatchMatMul,
                  "packed_batch_matmul": OpType.PackedBatchMatMul,
                  "dense": OpType.Dense,
                  "packed_dense": OpType.PackedDense,
                  "conv2d": OpType.Conv2d,
                  "cumsum": OpType.CumSum,
                  "softmax": OpType.Softmax}
  if op_str not in op_type_dict:
    raise ValueError("Unsupported op " + op_str)
  return op_type_dict[op_str]


def string_to_tuner_type(tuner_str):
  tuner_type_dict = {"autotvm": TunerType.AutoTVM,
                     "autoscheduler": TunerType.AutoScheduler,
                     "fast": TunerType.Fast}
  if tuner_str not in tuner_type_dict:
    raise ValueError("Unsupported tuner " + tuner_str)
  return tuner_type_dict[tuner_str]
