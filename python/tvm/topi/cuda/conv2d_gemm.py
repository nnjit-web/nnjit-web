
from tvm import te, target, autotvm
from ..arm_cpu.conv2d_int8 import _compute_conv2d_NHWC_quantized_without_transform
from ..webgpu.conv2d_gemm import _schedule_conv2d_NHWC


@autotvm.register_topi_compute("conv2d_NHWC_quantized_interleaved_without_transform.cuda")
def compute_conv2d_NHWC_quantized_interleaved_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    """Interface for interleaved compute_conv2d_NHWC_quantized_interleaved_without_transform"""
    return _compute_conv2d_NHWC_quantized_without_transform(
        cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels, True
    )


@autotvm.register_topi_schedule("conv2d_NHWC_quantized_interleaved_without_transform.cuda")
def schedule_conv2d_NHWC_quantized_interleaved_without_transform(cfg, outs):
    """Interface for interleaved schedule_conv2d_NHWC_quantized_interleaved"""
    return _schedule_conv2d_NHWC(cfg, outs, True)
