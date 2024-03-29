
import logging

# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re

from tvm import relay, topi
from tvm.te import SpecializedCondition

from .. import op as _op
from .generic import *
from .x86 import wrap_compute_conv2d_gemm, wrap_compute_conv2d_gemm_without_transform


@dense_strategy.register("webgpu")
def dense_strategy_webgpu(attrs, inputs, out_type, target):
    """dense webgpu strategy"""
    strategy = _op.OpStrategy()
    data, weights = inputs
    b, i = get_const_tuple(data.shape)
    o, _ = get_const_tuple(weights.shape)

    '''
    strategy.add_implementation(
        wrap_compute_dense(topi.gpu.dense_small_batch),
        wrap_topi_schedule(topi.webgpu.schedule_dense_small_batch),
        name="dense_small_batch.webgpu",
    )
    '''

    with SpecializedCondition(b >= 32):
        strategy.add_implementation(
            wrap_compute_dense(topi.gpu.dense_large_batch),
            wrap_topi_schedule(topi.webgpu.schedule_dense_large_batch),
            name="dense_large_batch.webgpu",
            plevel=5,
        )
    return strategy


@batch_matmul_strategy.register("webgpu")
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


@conv2d_strategy.register("webgpu")
def conv2d_strategy_webgpu(attrs, inputs, out_type, target):
    """conv2d webgpu strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    
    if groups == 1:
        if layout == "NCHW":
            if kernel_layout == "OIHW":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
                    name="conv2d_nchw.webgpu",
                )
        elif layout == "NHWC":
            if kernel_layout == "HWIO":
                strategy.add_implementation(
                    wrap_compute_conv2d_gemm(topi.webgpu.compute_conv2d_NHWC_native),
                    wrap_topi_schedule(topi.webgpu.schedule_conv2d_NHWC_native),
                    name="conv2d_NHWC_native.webgpu",
                )
                '''
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.gpu.conv2d_nhwc),
                    wrap_topi_schedule(topi.gpu.schedule_conv2d_nhwc),
                    name="conv2d_nhwc.gpu",
                )
                '''
    return strategy


@conv2d_gemm_strategy.register("webgpu")
def conv2d_gemm_strategy_webgpu(attrs, inputs, out_type, target):
    """conv2d_gemm webgpu strategy"""
    #import sys
    #print("tvm.relay.op.strategy.x86.conv2d_gemm_strategy_webgpu", file=sys.stderr)
    layout = attrs.data_layout
    data = inputs[0]
    strategy = _op.OpStrategy()

    native_compute = topi.webgpu.compute_conv2d_NHWC_native
    interleaved_compute = topi.webgpu.compute_conv2d_NHWC_interleaved
    if layout == "NHWC" and data.dtype in ["int8", "uint8", "float32"]:
        strategy.add_implementation(
            wrap_compute_conv2d_gemm(native_compute),
            wrap_topi_schedule(
                topi.webgpu.schedule_conv2d_NHWC_native
            ),
            name="conv2d_NHWC_native.webgpu",
        )
        '''
        strategy.add_implementation(
            wrap_compute_conv2d_gemm(interleaved_compute),
            wrap_topi_schedule(
                topi.webgpu.schedule_conv2d_NHWC_interleaved
            ),
            name="conv2d_NHWC_interleaved.webgpu",
        )
        '''
    else:
        raise RuntimeError(
            "Unsupported conv2d_NHWC layout {0}"
            "with datatype {1}".format(layout, data.dtype)
        )
    return strategy


@conv2d_gemm_without_weight_transform_strategy.register("webgpu")
def conv2d_gemm_without_weight_transform_strategy_webgpu(attrs, inputs, out_type, target):
    """Conv2dGemmWithoutWeightTransform webgpu strategy."""
    layout = attrs.data_layout
    data = inputs[0]
    strategy = _op.OpStrategy()
    interleaved_compute = topi.webgpu.compute_conv2d_NHWC_interleaved_without_transform
    #native_compute = topi.webgpu.compute_conv2d_NHWC_native_without_transform
    if layout == "NHWC" and data.dtype in ["int8", "uint8", "float32"]:
        strategy.add_implementation(
            wrap_compute_conv2d_gemm_without_transform(interleaved_compute),
            wrap_topi_schedule(
                topi.webgpu.schedule_conv2d_NHWC_interleaved_without_transform
            ),
            name="conv2d_NHWC_interleaved_without_transform.webgpu",
        )
    else:
        raise RuntimeError(
            "Unsupported conv2d_NHWC_without_transform layout {0}"
            "with datatype {1}".format(layout, data.dtype)
        )
    return strategy
