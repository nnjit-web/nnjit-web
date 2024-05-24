
import logging

# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re

from tvm import relay, topi
from tvm.te import SpecializedCondition

from .. import op as _op
from .generic import *


@batch_matmul_strategy.register(["webgpu", "gpu"])
def batch_matmul_strategy_webgpu(attrs, inputs, out_type, target):
    """BatchMatmul webgpu strategy."""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.webgpu.batch_matmul, need_out_dtype=True),
        wrap_topi_schedule(topi.webgpu.schedule_batch_matmul),
        name="batch_matmul.webgpu",
        plevel=10,
    )
    return strategy
