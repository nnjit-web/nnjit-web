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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""GEMM Convolution schedule on ARM"""
import tvm
from tvm import te
from tvm.topi import nn
from tvm.autotvm.task.space import AnnotateEntity, ReorderEntity, OtherOptionEntity
from ..utils import get_const_tuple, get_const_int
from ..nn.utils import get_pad_tuple
from .tensor_intrin import (
    gemm_4x4_int8_int8_int32,
    gemm_acc_4x4_int8_int8_int32,
    gemm_acc_nx16_int8_int8_int32,
    gemm_acc_2x2_int8_int8_int32,
)
from .arm_utils import is_aarch64_arm, is_dotprod_available, is_mmla_available
from ..utils import get_os_env_var_bool, get_os_env_var_int


def sleep(duration):
    import time
    time.sleep(duration)


def configure_knobs(cfg, M, K):
    """Configure auto-tuning knobs for the interleaved strategy"""

    x, y = cfg.axis(M // 4), cfg.axis(K // 16)
    cfg.define_reorder("reorder_gemm", [x, y], policy="candidate", candidate=[[x, y], [y, x]])

    outer_loop, inner_loop = cfg.axis(4), cfg.axis(16)
    cfg.define_annotate(
        "A_interleaved_unroll_vec", [outer_loop, inner_loop], policy="try_unroll_vec"
    )

    # Fallback configuration
    if cfg.is_fallback:
        cfg["reorder_gemm"] = ReorderEntity([0, 1])
        cfg["A_interleaved_unroll_vec"] = AnnotateEntity(["unroll", "vec"])

    if not is_dotprod_available():
        cfg.define_knob("gemm_quantized_unroll", [True, False])
        if cfg.is_fallback:
            cfg["gemm_quantized_unroll"] = OtherOptionEntity(False)


def my_configure_knobs(cfg, M, K):
    """Configure auto-tuning knobs for the interleaved strategy"""

    return

    #x, y = cfg.axis(M // 4), cfg.axis(K // 16)
    #cfg.define_reorder("reorder_gemm", [x, y], policy="candidate", candidate=[[x, y]])

    outer_loop, inner_loop = cfg.axis(4), cfg.axis(16)
    cfg.define_annotate(
        "A_interleaved_unroll_vec", [outer_loop, inner_loop], policy="try_unroll_vec"
    )

    # Fallback configuration
    if cfg.is_fallback:
        cfg["reorder_gemm"] = ReorderEntity([0, 1])
        cfg["A_interleaved_unroll_vec"] = AnnotateEntity(["unroll", "vec"])

    if not is_dotprod_available():
        cfg.define_knob("gemm_quantized_unroll", [False])
        if cfg.is_fallback:
            cfg["gemm_quantized_unroll"] = OtherOptionEntity(False)

# Compute function
def compute_conv2d_gemm_without_weight_transform(
    cfg,
    data,
    B_interleaved_t,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size,
    output_channels,
    interleave_A,
):
    """Compute conv2d by transforming the input,
    executing GEMM and transforming the output back"""
    #print("tvm.topi.arm_cpu.conv2d_gemm.compute_conv2d_gemm_without_weight_transform")
    batches, IH, IW, IC = get_const_tuple(data.shape)
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: IH %d, IW %d, IC %d" % (IH, IW, IC))

    KH, KW = get_const_tuple(kernel_size)
    OC = get_const_int(output_channels)
    kernel_area = KH * KW

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
    if pad_top or pad_left:
        data_pad = nn.pad(
            data, [0, pad_top, pad_left, 0], [0, pad_down, pad_right, 0], name="data_pad"
        )
    else:
        data_pad = data
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: OH %d, OW %d, OC %d" % (OH, OW, OC))

    # Im2col
    M = OH * OW
    K = IC * kernel_area
    N = OC
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: M %d, K %d, N %d" % (M, K, N))

    A_shape = (batches, M, K)
    if kernel_area == 1:
        A = tvm.topi.reshape(data_pad, A_shape)
    else:
        A = te.compute(
            A_shape,
            lambda n, x, y: data_pad[
                n,
                HSTR * (x // OW) + dilation_h * ((y // IC) // KW),
                WSTR * (x % OW) + dilation_w * ((y // IC) % KW),
                y % IC,
            ],
            name="data_im2col",
        )

    #  Pad if necessary
    use_my_compute = True
    if not use_my_compute:
        N_transformed = B_interleaved_t.shape[0]
        tile_rows_B = B_interleaved_t.shape[2]  # N
        tile_cols_B = B_interleaved_t.shape[3]  # K
    else:
        N_transformed = B_interleaved_t.shape[1]
        tile_rows_B = B_interleaved_t.shape[2]  # K
        tile_cols_B = B_interleaved_t.shape[3]  # N
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: N_transformed %d, tile_rows_B %d, tile_cols_B %d" % (N_transformed, tile_rows_B, tile_cols_B))

    # Select the tiling strategy for A.
    # The tiling information is chosen to maximize register usage during
    # the tile computation.
    #
    # Please refer to:
    # - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-performance-for-armv8-architectures # pylint: disable=line-too-long
    # - https://discuss.tvm.apache.org/t/rfc-accelerate-quantized-convolution-through-dot-product
    # - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-through-mmla-instruction
    # - Conv2DGemmWeightTransformRel in src/relay/op/nn/convolution.h
    # In order to have more information
    #
    if is_mmla_available():
        # If smmla/ummla is enabled, we are loading 8 rows from A. Each row
        # will contain 8 elements
        tile_rows_A = 8
        tile_cols_A = 8
    elif is_dotprod_available() and interleave_A:
        # If dot product has been enabled, and we are interleaving A
        # tile size should be 8x4
        tile_rows_A = 8
        tile_cols_A = 4
    else:
        # If either there is no dot product or if we are using a native strategy
        # tile size should be 4x16
        #tile_rows_A = 4
        #tile_cols_A = 16
        
        use_simd = True
        if not use_simd:
            tile_rows_A = 4
        else:
            #tile_rows_A = 5
            #tile_rows_A = 4
            import os
            tile_rows_A = os.getenv("TVM_GEMM_MB")
            tile_rows_A = 4 if tile_rows_A is None else int(tile_rows_A)
        tile_cols_A = tile_rows_B

    pad_M = 0
    pad_K = 0

    if M % tile_rows_A != 0:
        pad_M = tile_rows_A - (M % tile_rows_A)

    if K % tile_cols_A != 0:
        pad_K = tile_cols_A - (K % tile_cols_A)

    M_padded = M + pad_M
    K_padded = K + pad_K
    if not use_my_compute:
        N_padded = N_transformed * tile_rows_B
    else:
        N_padded = N_transformed * tile_cols_B
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: M_padded %d, K_padded %d, N_padded %d" % (M_padded, K_padded, N_padded))

    pad_before = (0, 0, 0)
    pad_after = (0, pad_M, pad_K)

    if pad_M != 0 or pad_K != 0:
        A = nn.pad(A, pad_before=pad_before, pad_after=pad_after, name="A_padded")

    idxm = tvm.tir.indexmod
    k = te.reduce_axis((0, K_padded), "k")

    if interleave_A:
        # Configuration space
        #configure_knobs(cfg, M_padded, K_padded)
        my_configure_knobs(cfg, M_padded, K_padded)

        # Pack the input data
        if not use_my_compute:
            A_interleaved = te.compute(
                (batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A),
                lambda b, x, y, z, w: A[b, z + tile_rows_A * x, w + tile_cols_A * y],
                name="A_interleaved",
            )
        else:
            have_input_transform = get_os_env_var_bool("TVM_CONV2D_GEMM_INPUT_TRANSFORM", True)
            if not have_input_transform:
                #A = te.placeholder((batches, M_padded, K_padded), name="A")
                A_interleaved = te.placeholder((batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A), name="A_interleaved")
            else:
                A_interleaved = te.compute(
                    (batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A),
                    lambda b, x, y, z, w: A[b, z + tile_rows_A * x, w + tile_cols_A * y],
                    name="A_interleaved",
                )
        if is_mmla_available():
            # Execute GEMM. In the case of mmla, we need to enforce the tiling
            # from the compute. This is because mmla is doing a tiled computation
            # as well. So we have a big 8x12 tile, with small 2x2 sub-tiles
            # generated by mmla. In theory we could make the tile 2x2 and
            # fuse and split during scheduling, but this would not work
            # because of possible padding
            C_interleaved = te.compute(
                (
                    batches,
                    M_padded // tile_rows_A,
                    N_transformed,
                    tile_rows_A // 2,
                    tile_rows_B // 2,
                    2,
                    2,
                ),
                lambda b, x, y, w, z, s, t: te.sum(
                    A_interleaved[b, x, k // tile_cols_A, 2 * w + s, idxm(k, tile_cols_A)].astype(
                        "int32"
                    )
                    * B_interleaved_t[y, k // tile_cols_B, 2 * z + t, idxm(k, tile_cols_B)].astype(
                        "int32"
                    ),
                    axis=k,
                ),
                name="C_interleaved",
            )
            # Unpack the result
            C = te.compute(
                (batches, M, N),
                lambda b, x, y: C_interleaved[
                    b,
                    x // tile_rows_A,
                    y // tile_rows_B,
                    idxm(x, tile_rows_A) // 2,
                    idxm(y, tile_rows_B) // 2,
                    idxm(idxm(x, tile_rows_A), 2),
                    idxm(idxm(y, tile_rows_B), 2),
                ].astype(out_dtype),
                name="C",
            )
        else:
            # Execute GEMM
            #gemm_dtype = "int32"
            gemm_dtype = out_dtype
            if not use_my_compute:
                C_interleaved = te.compute(
                    (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_rows_B),
                    lambda b, x, y, w, z: te.sum(
                        A_interleaved[b, x, k // tile_cols_A, w, idxm(k, tile_cols_A)].astype(gemm_dtype)
                        * B_interleaved_t[y, k // tile_cols_B, z, idxm(k, tile_cols_B)].astype(gemm_dtype),
                        axis=k,
                    ),
                    name="C_interleaved",
                )
            else:
                have_compute = True
                if not have_compute:
                    return A_interleaved
                if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
                    print("conv2d_gemm.py: b %d, x %d, y %d, w %d, z %d" % (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_cols_B))
                have_compute_sum = True
                if have_compute_sum:
                    C_interleaved = te.compute(
                        (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_cols_B),
                        lambda b, x, y, w, z: te.sum(
                            A_interleaved[b, x, k // tile_cols_A, w, idxm(k, tile_cols_A)].astype(gemm_dtype)
                            * B_interleaved_t[k // tile_rows_B, y, idxm(k, tile_rows_B), z].astype(gemm_dtype),
                            axis=k,
                        ),
                        name="C_interleaved",
                    )
                else:
                    C_interleaved = te.compute(
                        (batches, M_padded // tile_rows_A, N_transformed, K_padded, tile_rows_A, tile_cols_B),
                        lambda b, x, y, k, w, z: 
                            A_interleaved[b, x, k // tile_cols_A, w, idxm(k, tile_cols_A)].astype(gemm_dtype)
                            * B_interleaved_t[k // tile_rows_B, y, idxm(k, tile_rows_B), z].astype(gemm_dtype)
                        ,
                        name="C_interleaved",
                    )

                have_output_transform = get_os_env_var_bool("TVM_CONV2D_GEMM_OUTPUT_TRANSFORM", True)
                if not have_output_transform:
                    return C_interleaved
            # Unpack the result
            if not use_my_compute:
                C = te.compute(
                    (batches, M, N),
                    lambda b, x, y: C_interleaved[
                        b,
                        x // tile_rows_A,
                        y // tile_rows_B,
                        idxm(x, tile_rows_A),
                        idxm(y, tile_rows_B),
                    ].astype(out_dtype),
                    name="C",
                )
            else:
                C = te.compute(
                    (batches, M, N),
                    lambda b, x, y: C_interleaved[
                        b,
                        x // tile_rows_A,
                        y // tile_cols_B,
                        idxm(x, tile_rows_A),
                        idxm(y, tile_cols_B),
                    ].astype(out_dtype),
                    name="C",
                )
        zero = tvm.tir.const(0)
    else:
        # No need to pack/unpack, execute GEMM directly
        #gemm_dtype = "int32"
        gemm_dtype = out_dtype
        C = te.compute(
            (batches, M_padded, N_padded),
            lambda b, x, y: te.sum(
                A[b, x, k].astype(gemm_dtype)
                * B_interleaved_t[
                    y // tile_rows_B, k // tile_cols_B, idxm(y, tile_rows_B), idxm(k, tile_cols_B)
                ].astype(gemm_dtype),
                axis=k,
            ),
            name="C",
        )

        # We need to ensure that infer bound pass does not remove the padding
        # which is necessary for the tensorizations to work. So we need to
        # add a dummy reference to the padding area of the result
        zero = (
            tvm.tir.const(1, C.dtype) * C[0, M_padded - 1, N_padded - 1]
            - tvm.tir.const(1, C.dtype) * C[0, M_padded - 1, N_padded - 1]
        )

    # Reshape the result into a convolution output
    out_shape = (batches, OH, OW, OC)
    out = te.compute(
        out_shape,
        lambda b, x, y, z: (C(b, y + OW * x, z) + zero).astype(out_dtype),
        name="conv2d_gemm_output",
    )
    return out


def compute_conv2d_igemm_without_weight_transform(
    cfg,
    data,
    B_interleaved_t,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size,
    output_channels,
    interleave_A,
):
    batches, IH, IW, IC = get_const_tuple(data.shape)

    KH, KW = get_const_tuple(kernel_size)
    OC = get_const_int(output_channels)
    kernel_area = KH * KW

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
    if pad_top or pad_left:
        data_pad = nn.pad(
            data, [0, pad_top, pad_left, 0], [0, pad_down, pad_right, 0], name="data_pad"
        )
    else:
        data_pad = data

    _, PIH, PIW, _ = get_const_tuple(data_pad)

    M = OH * OW
    K = IC * kernel_area
    N = OC

    N_transformed = B_interleaved_t.shape[1]  # N
    tile_rows_B = B_interleaved_t.shape[2]  # KB
    tile_cols_B = B_interleaved_t.shape[3]  # NB
    tile_rows_A = 4  # MB
    tile_cols_A = tile_rows_B  # KB

    pad_M = 0
    pad_K = 0
    if K % tile_cols_A != 0:
        #pad_K = tile_cols_A - (K % tile_cols_A)
        raise ValueError("K mod KB should be 0, not " + (K % tile_cols_A))
    if M % tile_rows_A != 0:
        raise ValueError("M mod MB should be 0, not " + (M % tile_rows_A))
    M_padded = M + pad_M
    K_padded = K + pad_K

    # Compute matmul
    if interleave_A:
        idxm = tvm.tir.indexmod
        k = te.reduce_axis((0, K_padded), "k")
        gemm_dtype = out_dtype
        C_interleaved = te.compute(
            (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_cols_B),
            lambda b, x, y, w, z: te.sum(
                data_pad[b, (x * tile_rows_A + w) // PIW, (x * tile_rows_A + w) % PIW, k // (K * K)].astype(gemm_dtype)
                * B_interleaved_t[k // tile_rows_B, y, idxm(k, tile_rows_B), z].astype(gemm_dtype),
                axis=k,
            ),
            name="C_interleaved",
        )

        # Unpack output
        C = te.compute(
            (batches, M, N),
            lambda b, x, y: C_interleaved[
                b,
                x // tile_rows_A,
                y // tile_cols_B,
                idxm(x, tile_rows_A),
                idxm(y, tile_cols_B),
            ].astype(out_dtype),
            name="C",
        )

        zero = tvm.tir.const(0)
    else:
        raise ValueError("We have not support the case that A is not interleaved")

    # Reshape the result into a convolution output
    out_shape = (batches, OH, OW, OC)
    out = te.compute(
        out_shape,
        lambda b, x, y, z: (C(b, y + OW * x, z) + zero).astype(out_dtype),
        name="conv2d_igemm_output",
    )
    return out


def schedule_conv2d_gemm_interleaved_myself(cfg, s, out, final_out):
    """Schedule the conv2d_gemm interleaved strategy"""
    #return s
    
    #print("tvm.topi.arm_cpu.conv2d_gemm.schedule_conv2d_gemm_interleaved")
    enable_input_transform = get_os_env_var_bool("TVM_CONV2D_GEMM_INPUT_TRANSFORM", True)
    enable_compute = True
    enable_compute_sum = True
    enable_cache_write = True
    enable_output_transform = get_os_env_var_bool("TVM_CONV2D_GEMM_OUTPUT_TRANSFORM", True)
    if enable_output_transform:
        C = out.op.input_tensors[0]
        C_interleaved = C.op.input_tensors[0]
    else:
        C = None
        C_interleaved = out
    if enable_compute:
        A_interleaved = C_interleaved.op.input_tensors[0]
        B_interleaved = C_interleaved.op.input_tensors[1]
    else:
        A_interleaved = out.op.input_tensors[0]
        B_interleaved = out.op.input_tensors[1]

    if enable_input_transform:
        # Input transform
        A_interleaved_input = A_interleaved.op.input_tensors[0]
        if A_interleaved_input.op.name == "A_padded":
            s[A_interleaved_input].compute_at(s[A_interleaved], A_interleaved.op.axis[3])
            s[A_interleaved_input].vectorize(A_interleaved_input.op.axis[2])
            s[A_interleaved_input].compute_inline()
            data_im2col = A_interleaved_input.op.input_tensors[0]
        else:
            data_im2col = A_interleaved_input

        b, m, n = data_im2col.op.axis
        if data_im2col.op.name == "data_im2col":
            b = b
            #n_outer, n_inner = s[data_im2col].split(n, 16)
            #n_outer, n_inner = s[data_im2col].split(n, 4)
            #s[data_im2col].unroll(n_outer)
            #s[data_im2col].vectorize(n_inner)
            # NOTE(fucheng): Disable parallel.
            #b_m_fused = s[data_im2col].fuse(b, m)
            #s[data_im2col].parallel(b_m_fused)
        else:
            b = b
            s[data_im2col].compute_inline()
            #s[data_im2col].compute_root()
            #print(data_im2col.op.axis)
            #sleep(10000)
            #n_outer, n_inner = s[data_im2col].split(n, 4)
            #s[data_im2col].vectorize(data_im2col.op.axis[-1])
        
        data_im2col = C_interleaved.op.input_tensors[0]
        b, xo, ko, xi, ki = data_im2col.op.axis
        s[data_im2col].vectorize(ki)
        s[data_im2col].unroll(xi)

    if not enable_compute:
        return s
    
    mb = A_interleaved.shape[3]
    kb = A_interleaved.shape[4]
    nb = B_interleaved.shape[3]

    #enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
    #mr = 4; kr = 4; nr = 4
    #mr = 56; kr = 32; nr = 56
    #mr_list = [4]
    #kr_list = [4]
    #nr_list = [4]
    #mr = get_os_env_var_int("TVM_GEMM_MR", 4); mr_list = [mr]
    #kr = get_os_env_var_int("TVM_GEMM_KB", 4); kr_list = [kr]
    #nr = get_os_env_var_int("TVM_GEMM_NR", 4); nr_list = [nr]
    mr_list = [4, 8, 16, 32]
    kr_list = [kb]
    nr_list = [4, 8, 16, 32]
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: mr_list %s" % str(mr_list))
        print("conv2d_gemm.py: kr_list %s" % str(kr_list))
        print("conv2d_gemm.py: nr_list %s" % str(nr_list))
    cfg.define_knob("mr", mr_list)
    #cfg.define_knob("kr", kr_list)
    cfg.define_knob("nr", nr_list)
    mr = cfg["mr"].val
    #kr = get_os_env_var_int("TVM_GEMM_KR", 4)
    #kr = cfg["kr"].val
    kr = kb
    nr = cfg["nr"].val
    #mr = get_os_env_var_int("TVM_GEMM_MR", 4)
    #kr = B_interleaved.shape[2]
    #nr = get_os_env_var_int("TVM_GEMM_NR", 4)

    #print("conv2d_gemm.py: mr %d, kr %d, nr %d" % (mr, kr, nr))
    #f = open("logs/packed-batch-matmul-autotuning-configs-wasm.txt", "a")
    #f.write("mb %d, kb %d, nb %d, mr %d, kr %d, nr %d\n" % (mb, kb, nb, mr, kr, nr))
    #f.close()

    if enable_compute_sum:
        simd_size = 4
        enable_unroll = True
        if simd_size == 1 or not enable_unroll:
            enable_cache_write = False
        # Computation(through tensorize)
        b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]

        if enable_cache_write:
            CC = s.cache_write(C_interleaved, "local")
        
        #outer_gemm, inner_gemm = cfg["reorder_gemm"].apply(s, C_interleaved, [xo, yo])
        outer_gemm, inner_gemm = xo, yo
        #s[C_interleaved].reorder(b, xo, yo, k, xi, yi)
        #s[C_interleaved].reorder(b, xo, yo, xi, k, yi)
        #s[C_interleaved].reorder(b, xo, yo, xi, yi, k)

        enable_split_xi = True
        enable_split_yi = True
        if enable_split_xi and enable_split_yi:
            xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)
            if enable_unroll:
                s[C_interleaved].unroll(xii)
        
        if not enable_split_yi:
            s[C_interleaved].vectorize(yi)
        else:
            #yio, yii = s[C_interleaved].split(yi, nr)
            s[C_interleaved].reorder(b, xo, yo, xio, yio, xii, yii)
            if simd_size > 1:
                yiio, yiii = s[C_interleaved].split(yii, simd_size)
                if enable_unroll:
                    s[C_interleaved].unroll(yiio)
                s[C_interleaved].vectorize(yiii)
            else:
                if enable_unroll:
                    s[C_interleaved].unroll(yii)
            
            #b_outer_gemm_fused = s[C_interleaved].fuse(b, outer_gemm)
            
            if not enable_cache_write:
                k = C_interleaved.op.reduce_axis[0]
                ko, ki = s[C_interleaved].split(k, kr)
                if enable_split_xi:
                    #s[C_interleaved].reorder(b, xo, yo, ko, xio, xii, yio, yii, ki)
                    #s[C_interleaved].reorder(b, xo, yo, ko, yio, yii, xio, xii, ki)
                    #s[C_interleaved].reorder(b, xo, yo, ko, xio, yio, xii, yii, ki)
                    if simd_size > 1:
                        s[C_interleaved].reorder(b, xo, yo, xio, ko, ki, yio, xii, yiio, yiii)
                    else:
                        s[C_interleaved].reorder(b, xo, yo, xio, ko, ki, yio, xii, yii)
                else:
                    s[C_interleaved].reorder(b, xo, yo, ko, xi, yio, yii, ki)
            else:
                s[CC].compute_at(s[C_interleaved], yio)
                b, xo, yo, xi, yi = s[CC].op.axis
                xio, yio, xii, yii = s[CC].tile(xi, yi, x_factor=mr, y_factor=nr)
                k = CC.op.reduce_axis[0]
                ko, ki = s[CC].split(k, kr)
                s[CC].reorder(b, xo, yo, ko, ki, xio, yio, xii, yii)
                if simd_size > 1:
                    yiio, yiii = s[CC].split(yii, simd_size)
                    if enable_unroll:
                        s[CC].unroll(xii)
                        s[CC].unroll(yiio)
                    s[CC].vectorize(yiii)
                else:
                    if enable_unroll:
                        s[CC].unroll(xii)
                        s[CC].unroll(yii)
        
        # NOTE(fucheng): Disable parallel.
        #s[C_interleaved].parallel(b_outer_gemm_fused)
        if enable_input_transform:
            s[A_interleaved].compute_at(s[C_interleaved], b_outer_gemm_fused)
            _, _, _, outer_A_interleaved, inner_A_interleaved = A_interleaved.op.axis
            #cfg["A_interleaved_unroll_vec"].apply(
            #    s, A_interleaved, [outer_A_interleaved, inner_A_interleaved]
            #)
    else:
        b, xo, yo, k, xi, yi = C_interleaved.op.axis[0:6]
        b_outer_gemm_fused = s[C_interleaved].fuse(b, xo)
        if enable_input_transform:
            s[A_interleaved].compute_at(s[C_interleaved], b_outer_gemm_fused)
            s[C_interleaved].unroll(xi)
            yio, yii = s[C_interleaved].split(yi, 4)
            s[C_interleaved].unroll(yio)
            s[C_interleaved].vectorize(yii)

    if C is None:
        return s

    in_type = A_interleaved.dtype
    out_type = C.dtype

    k = C_interleaved.op.reduce_axis[0]
    _, M, N = C.shape
    if in_type in ["int8", "uint8"]:
        if is_mmla_available():
            #print("scheduling for mmla")
            gemm_acc = gemm_acc_2x2_int8_int8_int32(in_type)
            xi_inner, yi_inner = C_interleaved.op.axis[-2:]
            k_outer, k_inner = s[C_interleaved].split(k, 8)
            s[C_interleaved].reorder(
                b_outer_gemm_fused, inner_gemm, k_outer, xi, yi, xi_inner, yi_inner, k_inner
            )
            s[C_interleaved].tensorize(xi_inner, gemm_acc)
            s[C_interleaved].unroll(xi)
            s[C_interleaved].unroll(yi)
        elif is_dotprod_available():
            #print("scheduling for dotprod")
            gemm_acc = gemm_acc_4x4_int8_int8_int32(in_type)
            xi_outer, yi_outer, xi_inner, yi_inner = s[C_interleaved].tile(
                xi, yi, x_factor=8, y_factor=4
            )
            k_outer, k_inner = s[C_interleaved].split(k, 4)
            xi_inner_outer, xi_inner_inner = s[C_interleaved].split(xi_inner, 4)
            s[C_interleaved].reorder(
                b_outer_gemm_fused,
                inner_gemm,
                xi_outer,
                yi_outer,
                k_outer,
                xi_inner_outer,
                xi_inner_inner,
                yi_inner,
                k_inner,
            )
            s[C_interleaved].tensorize(xi_inner_inner, gemm_acc)
            s[C_interleaved].unroll(xi_inner_outer)

        elif is_aarch64_arm():
            #print("scheduling for aarch64")
            s[C_interleaved].reorder(yi, xi)
            K = A_interleaved_input.shape[2]
            assert in_type in ["int8", "uint8"], "Only int8 and uint8 gemm are supported"
            unroll = cfg["gemm_quantized_unroll"].val
            gemm = gemm_4x4_int8_int8_int32(M, N, K, unroll, in_type)
            s[C_interleaved].tensorize(yi, gemm)
        #else:
        #    print("no scheduling")

    if enable_cache_write and False:
        C = out.op.input_tensors[0]
        CC = C.op.input_tensors[0]
        print("s[out].op.name {}, s[out].op.axis {}".format(s[out].op.name, s[out].op.axis))
        print("s[C].op.name {}, s[C].op.axis {}".format(s[C].op.name, s[C].op.axis))
        print("s[CC].op.name {}, s[CC].op.axis {}".format(s[CC].op.name, s[CC].op.axis))
        _, _, _, xi, yi = s[CC].op.axis
        s[CC].unroll(xi)
        s[CC].vectorize(yi)
        #yio, yii = s[CC].split(yi, 4)
        #s[CC].unroll(yio)
        #s[CC].vectorize(yii)
        #exit(0)

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[C].compute_at(s[out], inner)
        s[out].vectorize(inner)
    
    if enable_output_transform:
        s[C].compute_inline()
        #s[out].compute_inline()
        n, h, w, c = out.op.axis
        outer, inner = s[out].split(c, nr)
        s[out].vectorize(inner)
        s[out].unroll(outer)
    
    return s


def schedule_conv2d_gemm_interleaved_autoscheduler_cache(cfg, s, out, final_out):
    use_cache_write = True
    
    conv2d_gemm_output = out
    C = out.op.input_tensors[0]
    C_interleaved = C.op.input_tensors[0]
    A_interleaved = C_interleaved.op.input_tensors[0]
    T_reshape = A_interleaved.op.input_tensors[0]

    T_reshape_ax0, T_reshape_ax1, T_reshape_ax2 = tuple(T_reshape.op.axis) + tuple(T_reshape.op.reduce_axis)
    A_interleaved_b, A_interleaved_x, A_interleaved_y, A_interleaved_z, A_interleaved_w = tuple(A_interleaved.op.axis) + tuple(A_interleaved.op.reduce_axis)
    C_interleaved_b, C_interleaved_x, C_interleaved_y, C_interleaved_w, C_interleaved_z, C_interleaved_k = tuple(C_interleaved.op.axis) + tuple(C_interleaved.op.reduce_axis)
    C_b, C_x, C_y = tuple(C.op.axis) + tuple(C.op.reduce_axis)
    conv2d_gemm_output_b, conv2d_gemm_output_x, conv2d_gemm_output_y, conv2d_gemm_output_z = tuple(conv2d_gemm_output.op.axis) + tuple(conv2d_gemm_output.op.reduce_axis)
    if use_cache_write:
        C_interleaved_local, = s.cache_write([C_interleaved], "local")
        C_interleaved_local_b_c, C_interleaved_local_x_c, C_interleaved_local_y_c, C_interleaved_local_w_c, C_interleaved_local_z_c, C_interleaved_local_k = tuple(C_interleaved_local.op.axis) + tuple(C_interleaved_local.op.reduce_axis)
        C_interleaved_local_b_c_o_i, C_interleaved_local_b_c_i = s[C_interleaved_local].split(C_interleaved_local_b_c, factor=1)
        C_interleaved_local_b_c_o_o_i, C_interleaved_local_b_c_o_i = s[C_interleaved_local].split(C_interleaved_local_b_c_o_i, factor=1)
        C_interleaved_local_b_c_o_o_o, C_interleaved_local_b_c_o_o_i = s[C_interleaved_local].split(C_interleaved_local_b_c_o_o_i, factor=1)
        C_interleaved_local_x_c_o_i, C_interleaved_local_x_c_i = s[C_interleaved_local].split(C_interleaved_local_x_c, factor=1)
        C_interleaved_local_x_c_o_o_i, C_interleaved_local_x_c_o_i = s[C_interleaved_local].split(C_interleaved_local_x_c_o_i, factor=98)
        C_interleaved_local_x_c_o_o_o, C_interleaved_local_x_c_o_o_i = s[C_interleaved_local].split(C_interleaved_local_x_c_o_o_i, factor=1)
        C_interleaved_local_y_c_o_i, C_interleaved_local_y_c_i = s[C_interleaved_local].split(C_interleaved_local_y_c, factor=1)
        C_interleaved_local_y_c_o_o_i, C_interleaved_local_y_c_o_i = s[C_interleaved_local].split(C_interleaved_local_y_c_o_i, factor=16)
        C_interleaved_local_y_c_o_o_o, C_interleaved_local_y_c_o_o_i = s[C_interleaved_local].split(C_interleaved_local_y_c_o_o_i, factor=1)
        C_interleaved_local_w_c_o_i, C_interleaved_local_w_c_i = s[C_interleaved_local].split(C_interleaved_local_w_c, factor=1)
        C_interleaved_local_w_c_o_o_i, C_interleaved_local_w_c_o_i = s[C_interleaved_local].split(C_interleaved_local_w_c_o_i, factor=1)
        C_interleaved_local_w_c_o_o_o, C_interleaved_local_w_c_o_o_i = s[C_interleaved_local].split(C_interleaved_local_w_c_o_o_i, factor=4)
        C_interleaved_local_z_c_o_i, C_interleaved_local_z_c_i = s[C_interleaved_local].split(C_interleaved_local_z_c, factor=8)
        C_interleaved_local_z_c_o_o_i, C_interleaved_local_z_c_o_i = s[C_interleaved_local].split(C_interleaved_local_z_c_o_i, factor=1)
        C_interleaved_local_z_c_o_o_o, C_interleaved_local_z_c_o_o_i = s[C_interleaved_local].split(C_interleaved_local_z_c_o_o_i, factor=1)
        C_interleaved_local_k_o, C_interleaved_local_k_i = s[C_interleaved_local].split(C_interleaved_local_k, factor=4)
        s[C_interleaved_local].reorder(C_interleaved_local_b_c_o_o_o, C_interleaved_local_x_c_o_o_o, C_interleaved_local_y_c_o_o_o, C_interleaved_local_w_c_o_o_o, C_interleaved_local_z_c_o_o_o, C_interleaved_local_b_c_o_o_i, C_interleaved_local_x_c_o_o_i, C_interleaved_local_y_c_o_o_i, C_interleaved_local_w_c_o_o_i, C_interleaved_local_z_c_o_o_i, C_interleaved_local_k_o, C_interleaved_local_b_c_o_i, C_interleaved_local_x_c_o_i, C_interleaved_local_y_c_o_i, C_interleaved_local_w_c_o_i, C_interleaved_local_z_c_o_i, C_interleaved_local_k_i, C_interleaved_local_b_c_i, C_interleaved_local_x_c_i, C_interleaved_local_y_c_i, C_interleaved_local_w_c_i, C_interleaved_local_z_c_i)
    C_interleaved_b_o, C_interleaved_b_i = s[C_interleaved].split(C_interleaved_b, factor=1)
    C_interleaved_x_o, C_interleaved_x_i = s[C_interleaved].split(C_interleaved_x, factor=98)
    C_interleaved_y_o, C_interleaved_y_i = s[C_interleaved].split(C_interleaved_y, factor=16)
    C_interleaved_w_o, C_interleaved_w_i = s[C_interleaved].split(C_interleaved_w, factor=4)
    C_interleaved_z_o, C_interleaved_z_i = s[C_interleaved].split(C_interleaved_z, factor=8)
    s[C_interleaved].reorder(C_interleaved_b_o, C_interleaved_x_o, C_interleaved_y_o, C_interleaved_w_o, C_interleaved_z_o, C_interleaved_b_i, C_interleaved_x_i, C_interleaved_y_i, C_interleaved_w_i, C_interleaved_z_i)
    if use_cache_write:
        s[C_interleaved_local].compute_at(s[C_interleaved], C_interleaved_z_o)
    s[conv2d_gemm_output].compute_root()
    s[C].compute_at(s[conv2d_gemm_output], conv2d_gemm_output_x)
    if use_cache_write:
        s[A_interleaved].compute_at(s[C_interleaved_local], C_interleaved_local_w_c_o_o_i)
    s[T_reshape].compute_at(s[A_interleaved], A_interleaved_x)
    C_interleaved_b_o_x_o_fused_y_o_fused = s[C_interleaved].fuse(C_interleaved_b_o, C_interleaved_x_o, C_interleaved_y_o)
    s[C_interleaved].parallel(C_interleaved_b_o_x_o_fused_y_o_fused)
    conv2d_gemm_output_b_x_fused = s[conv2d_gemm_output].fuse(conv2d_gemm_output_b, conv2d_gemm_output_x)
    s[conv2d_gemm_output].parallel(conv2d_gemm_output_b_x_fused)
    if use_cache_write:
        s[C_interleaved_local].pragma(C_interleaved_local_b_c_o_o_o, "auto_unroll_max_step", 64)
        s[C_interleaved_local].pragma(C_interleaved_local_b_c_o_o_o, "unroll_explicit", True)
        s[C_interleaved_local].vectorize(C_interleaved_local_z_c_i)
    s[C_interleaved].vectorize(C_interleaved_z_i)
    return s


def schedule_conv2d_gemm_interleaved_autoscheduler_nocache(cfg, s, out, final_out):
    conv2d_gemm_output = out
    C = out.op.input_tensors[0]
    C_interleaved = C.op.input_tensors[0]
    A_interleaved = C_interleaved.op.input_tensors[0]
    T_reshape = A_interleaved.op.input_tensors[0]

    T_reshape_ax0, T_reshape_ax1, T_reshape_ax2 = tuple(T_reshape.op.axis) + tuple(T_reshape.op.reduce_axis)
    A_interleaved_b, A_interleaved_x, A_interleaved_y, A_interleaved_z, A_interleaved_w = tuple(A_interleaved.op.axis) + tuple(A_interleaved.op.reduce_axis)
    C_interleaved_b, C_interleaved_x, C_interleaved_y, C_interleaved_w, C_interleaved_z, C_interleaved_k = tuple(C_interleaved.op.axis) + tuple(C_interleaved.op.reduce_axis)
    C_b, C_x, C_y = tuple(C.op.axis) + tuple(C.op.reduce_axis)
    conv2d_gemm_output_b, conv2d_gemm_output_x, conv2d_gemm_output_y, conv2d_gemm_output_z = tuple(conv2d_gemm_output.op.axis) + tuple(conv2d_gemm_output.op.reduce_axis)
    C_interleaved_b_o_i, C_interleaved_b_i = s[C_interleaved].split(C_interleaved_b, factor=1)
    C_interleaved_b_o_o_i, C_interleaved_b_o_i = s[C_interleaved].split(C_interleaved_b_o_i, factor=1)
    C_interleaved_b_o_o_o, C_interleaved_b_o_o_i = s[C_interleaved].split(C_interleaved_b_o_o_i, factor=1)
    C_interleaved_x_o_i, C_interleaved_x_i = s[C_interleaved].split(C_interleaved_x, factor=1)
    C_interleaved_x_o_o_i, C_interleaved_x_o_i = s[C_interleaved].split(C_interleaved_x_o_i, factor=28)
    C_interleaved_x_o_o_o, C_interleaved_x_o_o_i = s[C_interleaved].split(C_interleaved_x_o_o_i, factor=1)
    C_interleaved_y_o_i, C_interleaved_y_i = s[C_interleaved].split(C_interleaved_y, factor=1)
    C_interleaved_y_o_o_i, C_interleaved_y_o_i = s[C_interleaved].split(C_interleaved_y_o_i, factor=16)
    C_interleaved_y_o_o_o, C_interleaved_y_o_o_i = s[C_interleaved].split(C_interleaved_y_o_o_i, factor=1)
    C_interleaved_w_o_i, C_interleaved_w_i = s[C_interleaved].split(C_interleaved_w, factor=1)
    C_interleaved_w_o_o_i, C_interleaved_w_o_i = s[C_interleaved].split(C_interleaved_w_o_i, factor=2)
    C_interleaved_w_o_o_o, C_interleaved_w_o_o_i = s[C_interleaved].split(C_interleaved_w_o_o_i, factor=2)
    C_interleaved_z_o_i, C_interleaved_z_i = s[C_interleaved].split(C_interleaved_z, factor=8)
    C_interleaved_z_o_o_i, C_interleaved_z_o_i = s[C_interleaved].split(C_interleaved_z_o_i, factor=1)
    C_interleaved_z_o_o_o, C_interleaved_z_o_o_i = s[C_interleaved].split(C_interleaved_z_o_o_i, factor=1)
    C_interleaved_k_o, C_interleaved_k_i = s[C_interleaved].split(C_interleaved_k, factor=4)
    s[C_interleaved].reorder(C_interleaved_b_o_o_o, C_interleaved_x_o_o_o, C_interleaved_y_o_o_o, C_interleaved_w_o_o_o, C_interleaved_z_o_o_o, C_interleaved_b_o_o_i, C_interleaved_x_o_o_i, C_interleaved_y_o_o_i, C_interleaved_w_o_o_i, C_interleaved_z_o_o_i, C_interleaved_k_o, C_interleaved_b_o_i, C_interleaved_x_o_i, C_interleaved_y_o_i, C_interleaved_w_o_i, C_interleaved_z_o_i, C_interleaved_k_i, C_interleaved_b_i, C_interleaved_x_i, C_interleaved_y_i, C_interleaved_w_i, C_interleaved_z_i)
    s[C].compute_at(s[conv2d_gemm_output], conv2d_gemm_output_x)
    s[A_interleaved].compute_at(s[C_interleaved], C_interleaved_y_o_o_i)
    s[T_reshape].compute_at(s[A_interleaved], A_interleaved_w)
    C_interleaved_b_o_o_o_x_o_o_o_fused = s[C_interleaved].fuse(C_interleaved_b_o_o_o, C_interleaved_x_o_o_o)
    s[C_interleaved].parallel(C_interleaved_b_o_o_o_x_o_o_o_fused)
    conv2d_gemm_output_b_x_fused = s[conv2d_gemm_output].fuse(conv2d_gemm_output_b, conv2d_gemm_output_x)
    s[conv2d_gemm_output].parallel(conv2d_gemm_output_b_x_fused)
    s[C_interleaved].pragma(C_interleaved_b_o_o_o_x_o_o_o_fused, "auto_unroll_max_step", 512)
    s[C_interleaved].pragma(C_interleaved_b_o_o_o_x_o_o_o_fused, "unroll_explicit", True)
    s[C_interleaved].vectorize(C_interleaved_z_i)


def schedule_conv2d_gemm_interleaved(cfg, s, out, final_out):
    from .conv2d_gemm_schedule import schedule_conv2d_gemm_interleaved_tile
    from .conv2d_gemm_schedule import schedule_conv2d_gemm_interleaved_cache
    from .conv2d_gemm_schedule import schedule_conv2d_gemm_interleaved_vectorize_and_unroll
    import os
    schedule_name = os.getenv("TVM_TEST_SCHEDULE_NAME")
    if schedule_name == "tile":
        s = schedule_conv2d_gemm_interleaved_tile(cfg, s, out, final_out)
    elif schedule_name == "cache":
        s = schedule_conv2d_gemm_interleaved_cache(cfg, s, out, final_out)
    elif schedule_name == "vectorize" or schedule_name == "unroll":
        s = schedule_conv2d_gemm_interleaved_vectorize_and_unroll(cfg, s, out, final_out)
    else:
        # NOTE(fucheng): Do not forget to set use_auto_scheduler = True in conv2d_int8.py
        s = schedule_conv2d_gemm_interleaved_myself(cfg, s, out, final_out)
        #s = schedule_conv2d_gemm_interleaved_autoscheduler_cache(cfg, s, out, final_out)
        #s = schedule_conv2d_gemm_interleaved_autoscheduler_nocache(cfg, s, out, final_out)

    return s


def schedule_conv2d_gemm_native(cfg, s, out, final_out):
    """Schedule the conv2d_gemm hybrid strategy"""
    #print("tvm.topi.arm_cpu.conv2d_gemm.schedule_conv2d_gemm_native")
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    in_type = A.dtype

    # Computation
    b, x, y = C.op.axis
    (k,) = C.op.reduce_axis
    k_outer, k_inner = s[C].split(k, 16)
    x_outer, y_outer, x_inner, y_inner = s[C].tile(x, y, x_factor=4, y_factor=16)
    s[C].reorder(b, x_outer, y_outer, k_outer, x_inner, y_inner, k_inner)
    #gemm_acc = gemm_acc_nx16_int8_int8_int32(in_type, rows=1)
    s[C].unroll(x_inner)
    #s[C].tensorize(y_inner, gemm_acc)
    # NOTE(fucheng): Disable parallel.
    #s[C].parallel(x_outer)

    # Input transform
    if A.op.name == "A_padded":
        padding_A = True
        data_im2col = A.op.input_tensors[0]
    else:
        padding_A = False
        data_im2col = A

    b, m, n = data_im2col.op.axis
    if data_im2col.op.name == "data_im2col":
        n_outer, n_inner = s[data_im2col].split(n, 16)
        s[data_im2col].unroll(n_outer)
        s[data_im2col].vectorize(n_inner)
        # NOTE(fucheng): Disable parallel.
        #s[data_im2col].parallel(m)
    elif padding_A:
        s[data_im2col].compute_inline()
        s[A].compute_at(s[C], x_inner)
    else:
        s[data_im2col].compute_at(s[C], x_inner)

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[out].vectorize(inner)
    return s
