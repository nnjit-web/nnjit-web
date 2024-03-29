# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
# pylint: disable=no-value-for-parameter,import-outside-toplevel
"""Conv2D schedule on x86"""

import logging

import os
import tvm
from tvm import te
from tvm import autotvm
from tvm.contrib import dnnl
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from .. import nn
from ..generic import schedule_extern
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.utils import get_pad_tuple
from ..utils import get_const_tuple, traverse_inline
from ..utils import get_os_env_var_bool, get_os_env_var_int
from ..utils import build_2d_tile_sizes, build_3d_tile_sizes
from ..utils import build_antares_3d_tile_sizes, build_antares_cpu_4d_tile_sizes, build_antares_gpu_4d_tile_sizes
from . import conv2d_avx_1x1, conv2d_avx_common
from .conv2d_gemm import (
    compute_conv2d_gemm_without_weight_transform,
    schedule_conv2d_gemm_interleaved,
    schedule_conv2d_gemm_native,
)

logger = logging.getLogger("topi")


def _get_default_config(
    cfg, data, kernel, strides, padding, dilation, out_dtype, is_depthwise=False, layout="NCHW"
):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)
    if is_depthwise:
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype)
        from .depthwise_conv2d import _fallback_schedule

        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype, layout)
        is_kernel_1x1 = wkl.kernel_h == 1 and wkl.kernel_w == 1
        if is_kernel_1x1:
            conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        else:
            conv2d_avx_common._fallback_schedule(cfg, wkl)


@conv2d_infer_layout.register("cpu")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, _, dtype = workload
    batch_size, in_channel, in_height, in_width = data[1]
    out_channel, _, k_height, k_width = kernel[1]
    idxdiv = tvm.tir.indexdiv

    pt, pl, pb, pr = get_pad_tuple(padding, (k_height, k_width))
    hdilation, wdilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    dilated_kernel_h = (k_height - 1) * hdilation + 1
    dilated_kernel_w = (k_width - 1) * wdilation + 1
    out_height = idxdiv(in_height + pt + pb - dilated_kernel_h, strides[0]) + 1
    out_width = idxdiv(in_width + pl + pr - dilated_kernel_w, strides[1]) + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, idxdiv(in_channel, tile_ic), in_height, in_width, tile_ic)
    in_layout = "NCHW%dc" % tile_ic
    out_shape = (batch_size, idxdiv(out_channel, tile_oc), out_height, out_width, tile_oc)
    out_layout = "NCHW%dc" % tile_oc
    return ((in_shape, in_layout),), ((out_shape, out_layout),)


def schedule_conv2d_nhwc(outs):
    """Create schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    def _callback(op):
        if "conv2d_nhwc" in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, h_pad, w_pad, c_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, h, w, c = C.op.axis
            s[C].vectorize(c)

            O = output_op.output(0)
            if len(O.op.axis) == 4:  # schedule bias + bn + relu
                n, h, w, c = O.op.axis
                fused = s[O].fuse(n, h, w)
                s[O].parallel(fused)
                channels = int(O.shape[-1])
                if channels % 64 == 0:
                    c, ci = s[O].split(c, 64)
                    s[O].vectorize(ci)
                if C != O:
                    s[C].compute_at(s[O], c)

    traverse_inline(s, output_op, _callback)
    return s


def conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype):
    #import sys
    #print("tvm.topi.x86.conv2d_nchw", file=sys.stderr)
    layout = "NCHW"
    packed_out = conv2d_NCHWc(data, kernel, strides, padding, dilation, layout, layout, out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_conv2d_nchw(outs):
    """Create schedule for tensors"""
    return schedule_conv2d_NCHWc(outs)


def _pack_data(cfg, data, kernel):
    n, _, ih, iw = get_const_tuple(data.shape)
    oc, ic, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    ic_chunk = ic // ic_bn
    oc_chunk = oc // oc_bn

    # Handle dynamic shape to pass tuning dispatch.
    if isinstance(n, tvm.tir.Any):
        n = tvm.te.size_var("n")
    if isinstance(ih, tvm.tir.Any):
        ih = tvm.te.size_var("ih")
    if isinstance(iw, tvm.tir.Any):
        iw = tvm.te.size_var("iw")
    if isinstance(ic, tvm.tir.Any):
        raise RuntimeError("Dynamic input channel is not supported for conv2d.")

    data = te.compute(
        (n, ic_chunk, ih, iw, ic_bn),
        lambda bs, c, h, w, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec",
    )

    kernel = te.compute(
        (oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn),
        lambda occ, icc, k_h, k_w, icb, ocb: kernel[occ * oc_bn + ocb, icc * ic_bn + icb, k_h, k_w],
        name="kernel_vec",
    )

    return data, kernel


@autotvm.register_topi_compute("conv2d_NCHWc.x86")
def conv2d_NCHWc(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
            kernel.shape
        )
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kernel_height == 1 and kernel_width == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kernel_height, kernel_width))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    cfg.define_split("tile_ic", in_channel, num_outputs=2)
    cfg.define_split("tile_oc", num_filter, num_outputs=2)
    cfg.define_split(
        "tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64, policy="verbose"
    )
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder(
                (num_filter, in_channel, kernel_height, kernel_width), dtype=kernel.dtype
            ),
            strides,
            padding,
            dilation,
            out_dtype,
        )

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, in_channel // cfg["tile_ic"].size[-1], ih, iw, cfg["tile_ic"].size[-1])
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                num_filter // cfg["tile_oc"].size[-1],
                in_channel // cfg["tile_ic"].size[-1],
                kernel_height,
                kernel_width,
                cfg["tile_ic"].size[-1],
                cfg["tile_oc"].size[-1],
            )
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel")
        else:
            data, kernel = _pack_data(cfg, data, kernel)

    return nn.conv2d_NCHWc(data, kernel, strides, padding, dilation, layout, out_layout, out_dtype)


@autotvm.register_topi_schedule("conv2d_NCHWc.x86")
def schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            (
                _,
                _,
                kh,
                kw,
                _,
                _,
            ) = get_const_tuple(kernel_vec.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nchw_dnnl.x86")
def conv2d_nchw_dnnl(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d in NCHW format using dnnl."""
    groups = 1
    _out = dnnl.dnnl_conv2d(data, kernel, strides, padding, dilation, groups, False, out_dtype)
    return _out


@autotvm.register_topi_schedule("conv2d_nchw_dnnl.x86")
def schedule_conv2d_nchw_dnnl(_, outs):
    """Create schedule for conv2d_nchw_dnnl"""
    return schedule_extern(outs)


@autotvm.register_topi_compute("conv2d_nhwc_dnnl.x86")
def conv2d_nhwc_dnnl(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d in NHWC format using dnnl."""
    groups = 1
    _out = dnnl.dnnl_conv2d(data, kernel, strides, padding, dilation, groups, True, out_dtype)
    return _out


@autotvm.register_topi_schedule("conv2d_nhwc_dnnl.x86")
def schedule_conv2d_nhwc_dnnl(_, outs):
    """Create schedule for conv2d_nhwc_dnnl"""
    return schedule_extern(outs)


# FIXME - https://github.com/apache/tvm/issues/4122
# _declaration_conv_nhwc_pack expects kernel layout to be HWOI. However, the tests use HWIO
# layout. Commenting until we have clarity about the nhwc_pack implementation from the author.
# elif layout == 'NHWC' and kh == 1 and kw == 1 and kernel.dtype == "int8":
#     if cfg.is_fallback:
#         _get_default_config(cfg, data, kernel, strides, padding, out_dtype, False, layout)
#     # specialize for INT8 1X1 conv on X86
#     return conv2d_avx_1x1._declaration_conv_nhwc_pack(cfg, data, kernel, strides,
#                                                       padding, dilation, out_dtype)


def _compute_conv2d_NHWC(
    cfg, data, kernel, strides, padding, dilation, out_dtype, interleave_A
):
    N, IH, IW, IC = get_const_tuple(data.shape)
    KH, KW, _, OC = get_const_tuple(kernel.shape)
    #print("conv2d.py: data_shape {}, kernel_shape {}".format(data.shape, kernel.shape))

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = get_const_tuple(dilation)

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1

    MM = OH * OW
    MK = KH * KW * IC
    MN = OC
    
    enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
    if enable_tuning:
        space_name = os.getenv("TVM_TUNING_SPACE_NAME")
        if space_name is None or space_name == "default" or space_name == "large":
            cfg.define_split(
                "tile_y", 32 if isinstance(MM, (tvm.tir.Var, tvm.tir.Any)) else MN, num_outputs=3
            )
            if space_name == "default":
                cfg.define_split(
                    "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN, num_outputs=3
                )
            else:
                backend = os.getenv("BACKEND")
                if backend.find("wasm") >= 0:
                    cfg.define_split(
                        "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN, num_outputs=3,
                        filter=lambda x: x.size[-1] == 4
                    )
                else:
                    cfg.define_split(
                        "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN, num_outputs=3
                    )
            cfg.define_split(
                "tile_k", 32 if isinstance(MK, (tvm.tir.Var, tvm.tir.Any)) else MK, num_outputs=2
            )
            if space_name == "default":
                cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
            else:
                cfg.define_knob("auto_unroll_max_step", [64])
            cfg.define_knob("unroll_explicit", [1])
        else:
            mb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            kb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            nb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            mr_cadidates = [4, 8, 16, 32]
            nr_cadidates = [4, 8, 16, 32]
            backend = os.getenv("BACKEND")
            if backend.find("webgpu") >= 0:
                tile_b_sizes = [[-1, 1, 1, 1]]
                tile_y_sizes = build_antares_gpu_4d_tile_sizes(mb_cadidates, mr_cadidates)
                tile_x_sizes = build_antares_gpu_4d_tile_sizes(nb_cadidates, nr_cadidates)
                tile_k_sizes = build_antares_3d_tile_sizes(kb_cadidates)
                cfg.define_split(
                    "tile_b", 1 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N,
                    policy="candidate", num_outputs=4, candidate=tile_b_sizes
                )
                cfg.define_split(
                    "tile_y", 32 if isinstance(MM, (tvm.tir.Var, tvm.tir.Any)) else MM,
                    policy="candidate", num_outputs=4, candidate=tile_y_sizes
                )
                cfg.define_split(
                    "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN,
                    policy="candidate", num_outputs=4, candidate=tile_x_sizes
                )
                cfg.define_split(
                    "tile_k", 32 if isinstance(MK, (tvm.tir.Var, tvm.tir.Any)) else MK,
                    policy="candidate", num_outputs=3, candidate=tile_k_sizes
                )
                cfg.define_knob("vectorize", [0, 1])
            else:
                tile_y_sizes = build_3d_tile_sizes(mb_cadidates, mr_cadidates)
                tile_x_sizes = build_3d_tile_sizes(nb_cadidates, nr_cadidates)
                tile_k_sizes = build_2d_tile_sizes(kb_cadidates)
                cfg.define_split(
                    "tile_y", 32 if isinstance(MM, (tvm.tir.Var, tvm.tir.Any)) else MM,
                    policy="candidate", num_outputs=3, candidate=tile_y_sizes
                )
                cfg.define_split(
                    "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN,
                    policy="candidate", num_outputs=3, candidate=tile_x_sizes
                )
                cfg.define_split(
                    "tile_k", 32 if isinstance(MK, (tvm.tir.Var, tvm.tir.Any)) else MK,
                    policy="candidate", num_outputs=2, candidate=tile_k_sizes
                )
    else:
        space_name = os.getenv("TVM_TUNING_SPACE_NAME")
        if space_name is None or space_name == "default" or space_name == "large":
            tile_y = [1, 8, 4]
            tile_x = [1, 8, 4]
            tile_k = [1, 4]
            cfg.define_split(
                "tile_y", 32 if isinstance(MM, (tvm.tir.Var, tvm.tir.Any)) else MM,
                policy="candidate", num_outputs=3, candidate=[tile_y]
            )
            cfg.define_split(
                "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN,
                policy="candidate", num_outputs=3, candidate=[tile_x]
            )
            cfg.define_split(
                "tile_k", 32 if isinstance(MK, (tvm.tir.Var, tvm.tir.Any)) else MK,
                policy="candidate", num_outputs=2, candidate=[tile_k]
            )
            cfg.define_knob("auto_unroll_max_step", [64])
            cfg.define_knob("unroll_explicit", [1])
        else:
            backend = os.getenv("BACKEND")
            if backend.find("webgpu") >= 0:
                tile_y = [1, 8, 4, 1]
                tile_x = [1, 8, 4, 1]
                tile_k = [1, 1, 32]
                cfg.define_split(
                    "tile_y", 32 if isinstance(MM, (tvm.tir.Var, tvm.tir.Any)) else MM,
                    policy="candidate", num_outputs=4, candidate=[tile_y]
                )
                cfg.define_split(
                    "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN,
                    policy="candidate", num_outputs=4, candidate=[tile_x]
                )
                cfg.define_split(
                    "tile_k", 32 if isinstance(MK, (tvm.tir.Var, tvm.tir.Any)) else MK,
                    policy="candidate", num_outputs=3, candidate=[tile_k]
                )
                cfg.define_knob("vectorize", [0])
            else:
                tile_y = [1, 4, 8]
                tile_x = [1, 4, 8]
                tile_k = [1, 4]
                cfg.define_split(
                    "tile_y", 32 if isinstance(MM, (tvm.tir.Var, tvm.tir.Any)) else MM,
                    policy="candidate", num_outputs=3, candidate=[tile_y]
                )
                cfg.define_split(
                    "tile_x", 32 if isinstance(MN, (tvm.tir.Var, tvm.tir.Any)) else MN,
                    policy="candidate", num_outputs=3, candidate=[tile_x]
                )
                cfg.define_split(
                    "tile_k", 32 if isinstance(MK, (tvm.tir.Var, tvm.tir.Any)) else MK,
                    policy="candidate", num_outputs=2, candidate=[tile_k]
                )

    if cfg.is_fallback:
        cfg["tile_y"] = SplitEntity([-1, 8, 4])
        cfg["tile_x"] = SplitEntity([-1, 8, 4])
        cfg["tile_k"] = SplitEntity([-1, 4])
        cfg["auto_unroll_max_step"] = OtherOptionEntity(16)
        cfg["unroll_explicit"] = OtherOptionEntity(1)

    tile_rows_B, tile_cols_B = cfg["tile_k"].size[-1], (cfg["tile_x"].size[-2] * cfg["tile_x"].size[-1])

    #enable_tuning = autotvm.GLOBAL_SCOPE.in_tuning
    enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", True)
    if enable_tuning or True:
        #print("conv2d.py: disable kernel transformation (i.e., weight packing)")
        pad_K = 0
        pad_N = 0

        if MK % tile_rows_B != 0:
            pad_K = tile_rows_B - (MK % tile_rows_B)

        if MN % tile_cols_B != 0:
            pad_N = tile_cols_B - (MN % tile_cols_B)

        N_padded = MN + pad_N
        K_padded = MK + pad_K
        
        conv2d_packing = os.getenv("TVM_CONV2D_PACKING")
        if conv2d_packing is None or conv2d_packing == "none":
            kernel = te.placeholder((K_padded, N_padded), name="weight_transformed")
        elif conv2d_packing == "k":
            packw_bn = cfg["tile_x"].size[-1]
            packw_shape = (N_padded // packw_bn, K_padded, packw_bn)
            kernel = te.placeholder(packw_shape, name="weight_transformed")
        elif conv2d_packing == "kn":
            kernel = te.placeholder(
                (K_padded // tile_rows_B, N_padded // tile_cols_B, tile_rows_B, tile_cols_B),
                name="weight_transformed"
            )

    else:
        #print("conv2d.py: enable kernel transformation (i.e., weight packing)")
        kernel = nn.conv2d_gemm_weight_transform(cfg, kernel, tile_rows_B, tile_cols_B)
    return compute_conv2d_gemm_without_weight_transform(
        cfg, data, kernel, strides, padding, dilation, out_dtype, (KH, KW), OC, interleave_A
    )


def _compute_conv2d_NHWC_without_transform(
    cfg,
    data,
    B,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size=None,
    output_channels=None,
    interleave_A=False,
):
    return compute_conv2d_gemm_without_weight_transform(
        cfg,
        data,
        B,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        interleave_A,
    )


def _schedule_conv2d_NHWC(cfg, outs, interleave_A):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    # Vectorize the output and then inline all the rest
    out = outs[0]
    #n, h, w, c = out.op.axis
    #n_h_fused = s[out].fuse(n, h)
    #outer, inner = s[out].split(c, 4)
    #s[out].vectorize(inner)
    #s[out].parallel(n_h_fused)

    def _callback(op):
        """Traverse operators from computation graph"""
        if op.name in ["conv2d_gemm_output", "C", "C_interleaved"]:
            conv_out = op.output(0)
            if interleave_A:
                schedule_conv2d_gemm_interleaved(cfg, s, conv_out, out)
            else:
                schedule_conv2d_gemm_native(cfg, s, conv_out, out)

            '''
            if out != conv_out:
                s[conv_out].compute_at(s[out], inner)
            else:
                C = conv_out.op.input_tensors[0]
                if interleave_A:
                    s[C].compute_at(s[out], inner)
            '''

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_NHWC_interleaved.x86")
def compute_conv2d_NHWC_interleaved(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Interface for interleaved compute_conv2d_NHWC_interleaved"""
    return _compute_conv2d_NHWC(
        cfg, data, kernel, strides, padding, dilation, out_dtype, True
    )


@autotvm.register_topi_compute("conv2d_NHWC_interleaved_without_transform.x86")
def compute_conv2d_NHWC_interleaved_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    #import sys
    #print("tvm.topi.x86.conv2d_int8.compute_conv2d_NHWC_interleaved_without_transform", file=sys.stderr)
    """Interface for interleaved compute_conv2d_NHWC_interleaved_without_transform"""
    return _compute_conv2d_NHWC_without_transform(
        cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels, True
    )


@autotvm.register_topi_schedule("conv2d_NHWC_interleaved.x86")
def schedule_conv2d_NHWC_interleaved(cfg, outs):
    """Interface for interleaved schedule_conv2d_NHWC_interleaved"""
    return _schedule_conv2d_NHWC(cfg, outs, True)


@autotvm.register_topi_schedule("conv2d_NHWC_interleaved_without_transform.x86")
def schedule_conv2d_NHWC_interleaved_without_transform(cfg, outs):
    """Interface for interleaved schedule_conv2d_NHWC_interleaved"""
    return _schedule_conv2d_NHWC(cfg, outs, True)


@autotvm.register_topi_compute("conv2d_NHWC_native.x86")
def compute_conv2d_NHWC_native(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Interface for native compute_conv2d_NHWC"""
    return _compute_conv2d_NHWC(
        cfg, data, kernel, strides, padding, dilation, out_dtype, False
    )


@autotvm.register_topi_compute("conv2d_NHWC_native_without_transform.x86")
def compute_conv2d_NHWC_native_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    #import sys
    #print("tvm.topi.x86.conv2d_int8.compute_conv2d_NHWC_native_without_transform", file=sys.stderr)
    """Interface for compute_conv2d_NHWC_native_without_transform"""
    return _compute_conv2d_NHWC_without_transform(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        False,
    )


@autotvm.register_topi_schedule("conv2d_NHWC_native.x86")
def schedule_conv2d_NHWC_native(cfg, outs):
    """Interface for native schedule_conv2d_NHWC"""
    return _schedule_conv2d_NHWC(cfg, outs, False)


@autotvm.register_topi_schedule("conv2d_NHWC_native_without_transform.x86")
def schedule_conv2d_NHWC_native_without_transform(cfg, outs):
    """Interface for native schedule_conv2d_NHWC"""
    return _schedule_conv2d_NHWC(cfg, outs, False)
