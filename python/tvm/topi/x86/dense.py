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
# pylint: disable=invalid-name,too-many-locals,unused-variable
# pylint: disable=no-value-for-parameter
"""x86 dense operators"""
from __future__ import absolute_import as _abs
import os
import tvm
from tvm import te
from tvm import autotvm
from tvm.topi import nn
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas
from tvm.contrib import mkl
from tvm.contrib import dnnl

from .utils import get_simd_32bit_lanes
from .. import generic, tag
from ..utils import traverse_inline, get_const_tuple
from .tensor_intrin import dot_16x1x16_uint8_int8_int32_cascadelake
from ..utils import get_os_env_var_bool, get_os_env_var_int
from ..utils import build_2d_tile_sizes, build_3d_tile_sizes


DENSE_X86_DEFAULT_CONFIGS = {
    "do_fuse": True,
    "do_parallel": False,
    "cfg_define_type": "noknob"
}


def _schedule_dense_pack_template(cfg, s, C, O, do_parallel=False):
    space_name = os.getenv("TVM_TUNING_SPACE_NAME")
    if space_name is None or space_name == "default":
        schedule_type = "default"
    elif space_name == "large" or space_name == "small":
        schedule_type = "myself"
    #print("dense.py: cfg {}".format(cfg))
    #print("dense.py: space_name {}, schedule_type {}".format(space_name, schedule_type))
    #print("dense.py: s[C].op {}, s[O].op {}".format(s[C].op, s[O].op))

    if schedule_type == "default":
        do_fuse = True
        auto_unroll_max_step = -1

        A, packedB = s[C].op.input_tensors

        CC = s.cache_write(C, "global")
        y, x = s[C].op.axis
        (k,) = s[CC].op.reduce_axis

        yt, yo, yi = cfg["tile_y"].apply(s, C, y)
        xt, xo, xi = cfg["tile_x"].apply(s, C, x)
        s[C].reorder(xt, yt, yo, xo, yi, xi)
        if do_fuse:
            xyt = s[C].fuse(xt, yt)
            if C == O and do_parallel:
                s[C].parallel(xyt)
            xyo = s[C].fuse(yo, xo)
        if auto_unroll_max_step < 1:
            s[C].unroll(yi)
        else:
            s[C].pragma(yi, "auto_unroll_max_step", auto_unroll_max_step)
        s[C].vectorize(xi)

        if do_fuse:
            s[CC].compute_at(s[C], xyo)
        else:
            s[CC].compute_at(s[C], xo)
        y, x = s[CC].op.axis
        ko, ki = cfg["tile_k"].apply(s, CC, k)
        #s[CC].reorder(ko, ki, y, x)
        s[CC].reorder(ko, y, ki, x)
        s[CC].vectorize(x)

        tile_inner = cfg["tile_inner"].size[-1]
        if tile_inner > 1:
            yo, yi = s[CC].split(y, tile_inner)
            s[CC].reorder(ko, yo, ki, yi, x)
            if auto_unroll_max_step < 1:
                s[CC].unroll(yo)
                s[CC].unroll(ki)
                s[CC].unroll(yi)
            else:
                s[CC].pragma(yo, "auto_unroll_max_step", auto_unroll_max_step)
        else:
            if auto_unroll_max_step < 1:
                s[CC].unroll(ki)
                s[CC].unroll(y)
            else:
                s[CC].pragma(ki, "auto_unroll_max_step", auto_unroll_max_step)

        if C != O:
            y, x = s[O].op.axis
            yt, yo, yi = cfg["tile_y"].apply(s, O, y)
            xt, xo, xi = cfg["tile_x"].apply(s, O, x)
            s[O].reorder(xt, yt, yo, xo, yi, xi)
            if do_fuse:
                xyt = s[O].fuse(xt, yt)
                s[C].compute_at(s[O], xyt)
            else:
                s[C].compute_at(s[O], xt)
            s[O].vectorize(xi)
            if do_parallel:
                s[O].parallel(xyt)
    else:
        do_fuse = False
        do_cache_write = True
        
        A, packedB = s[C].op.input_tensors

        if do_cache_write:
            CC = s.cache_write(C, "local")
            (k,) = s[CC].op.reduce_axis
        else:
            (k,) = s[C].op.reduce_axis
        
        y, x = s[C].op.axis

        if space_name == "large":
            yt, yo, yi = cfg["tile_y"].apply(s, C, y)
            xt, xo, xi = cfg["tile_x"].apply(s, C, x)
            ko, ki = cfg["tile_k"].apply(s, CC, k)
            tile_inner = cfg["tile_inner"].size[-1]
            do_vectorize = cfg["tile_x"].size[-1] % 4 == 0
        elif space_name == "small":
            tile_inner = 1
            if DENSE_X86_DEFAULT_CONFIGS["cfg_define_type"] == "knob":
                do_vectorize = cfg["nr"].val % 4 == 0
                yo, xo, yi, xi = s[C].tile(y, x, x_factor=cfg["mr"].val, y_factor=cfg["nr"].val)
                yt, xt, yo, xo = s[C].tile(yo, xo, x_factor=cfg["mb"].val // cfg["mr"].val, y_factor=cfg["nb"].val // cfg["nr"].val)
                if do_cache_write:
                    ko, ki = s[CC].split(k, cfg["kb"].val)
                else:
                    ko, ki = s[C].split(k, cfg["kb"].val)
            else:
                do_vectorize = cfg["tile_x"].size[-1] % 4 == 0
                yt, yo, yi = cfg["tile_y"].apply(s, C, y)
                xt, xo, xi = cfg["tile_x"].apply(s, C, x)
                if do_cache_write:
                    ko, ki = cfg["tile_k"].apply(s, CC, k)
                else:
                    ko, ki = cfg["tile_k"].apply(s, C, k)

        if do_vectorize:
            xio, xii = s[C].split(xi, 4)
            if do_cache_write:
                s[C].reorder(yt, xt, yo, xo, yi, xio, xii)
            else:
                s[C].reorder(yt, xt, ko, yo, xo, ki, yi, xio, xii)
                s[C].unroll(ki)
            if do_fuse:
                xyt = s[C].fuse(xt, yt)
                xyo = s[C].fuse(yo, xo)
            s[C].vectorize(xii)
        else:
            s[C].reorder(yt, xt, yo, xo, yi, xi)
            if do_fuse:
                xyt = s[C].fuse(xt, yt)
                xyo = s[C].fuse(yo, xo)
        s[C].pragma(yi, "auto_unroll_max_step", 256)

        if do_fuse:
            s[CC].compute_at(s[C], xyo)
        else:
            if do_cache_write:
                s[CC].compute_at(s[C], xo)

        if do_cache_write:
            y, x = s[CC].op.axis
            
            if do_vectorize:
                xo, xi = s[CC].split(x, 4)
                s[CC].reorder(ko, ki, y, xo, xi)
                #s[CC].reorder(ko, y, ki, xo, xi)
                #s[CC].reorder(ko, y, xo, ki, xi)
                s[CC].vectorize(xi)
            else:
                s[CC].reorder(ko, ki, y, x)

        enable_inner_tile = False
        if enable_inner_tile and tile_inner > 1:
            yo, yi = s[CC].split(y, tile_inner)
            if do_vectorize:
                s[CC].reorder(ko, yo, ki, yi, xo, xi)
            else:
                s[CC].reorder(ko, yo, ki, yi, x)
            #s[CC].unroll(yo)
            #s[CC].unroll(ki)
            #s[CC].unroll(yi)

            #s[CC].pragma(yo, "auto_unroll_max_step", 64)
            s[CC].pragma(yi, "auto_unroll_max_step", 64)

            #s[CC].unroll(ko)
            #s[CC].unroll(ki)
            #s[CC].unroll(yo)
            #s[CC].unroll(yi)
            #s[CC].unroll(xo)
            #s[CC].pragma(ko, "auto_unroll_max_step", 64)
        else:
            if do_cache_write:
                #s[CC].unroll(ki)
                #s[CC].unroll(y)
                s[CC].pragma(y, "auto_unroll_max_step", 256)

        if C != O:
            y, x = s[O].op.axis
            xo, xi = s[O].split(x, 4)
            xoo, xoi = s[O].split(xo, 4)
            s[O].vectorize(xi)
            s[O].unroll(xoi)
    
    return s


def _schedule_dense_nopack_template(cfg, s, C, do_parallel=False):
    schedule_type = "myself"
    if schedule_type == "default":
        do_fuse = True
        
        y, x = s[C].op.axis
        (kk,) = s[C].op.reduce_axis
        yo, yi = cfg["tile_y"].apply(s, C, y)
        xo, xi = cfg["tile_x"].apply(s, C, x)
        s[C].reorder(yo, xo, yi, xi)
        if do_fuse:
            xyo = s[C].fuse(yo, xo)
            if do_parallel:
                s[C].parallel(xyo)
        #s[C].unroll(kk)
        s[C].pragma(kk, "auto_unroll_max_step", 64)

        (CC,) = s[C].op.input_tensors
        if do_fuse:
            s[CC].compute_at(s[C], xyo)
        else:
            s[CC].compute_at(s[C], xo)
        z, y, x = s[CC].op.axis
        (k,) = s[CC].op.reduce_axis
        if do_fuse:
            yz = s[CC].fuse(z, y)
        s[CC].reorder(k, yz, x)
        #s[CC].unroll(yz)
        s[CC].pragma(yz, "auto_unroll_max_step", 64)
        s[CC].vectorize(x)
    else:
        y, x = s[C].op.axis
        (kk,) = s[C].op.reduce_axis
        yo, yi = cfg["tile_y"].apply(s, C, y)
        xo, xi = cfg["tile_x"].apply(s, C, x)
        s[C].reorder(yo, xo, yi, xi)
        if cfg["tile_x"].size[-1] % 4 == 0:
            xio, xii = s[C].split(xi, 4)
            s[C].reorder(yo, xo, yi, xio, xii)
            s[C].vectorize(xii)

        (CC,) = s[C].op.input_tensors
        s[CC].compute_at(s[C], xo)
        z, y, x = s[CC].op.axis
        (k,) = s[CC].op.reduce_axis
        s[CC].reorder(k, z, y, x)
        if cfg["tile_x"].size[-1] % 4 == 0:
            xo, xi = s[CC].split(x, 4)
            s[CC].reorder(k, z, y, xo, xi)
            s[CC].vectorize(xi)
        else:
            s[CC].vectorize(x)
        s[CC].pragma(z, "auto_unroll_max_step", 64)
    
    return s


def _default_dense_pack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, (tvm.tir.Var, tvm.tir.Any)):
        M = 16
    if isinstance(N, (tvm.tir.Var, tvm.tir.Any)):
        N = 16
    if isinstance(K, (tvm.tir.Var, tvm.tir.Any)):
        K = 16

    vec_width = get_simd_32bit_lanes()
    tilex_ii = 1
    for bn in range(vec_width * 2, 0, -1):
        if N % bn == 0:
            tilex_ii = bn
            break
    NN = N // tilex_ii
    tilex_oi = 1
    while NN // tilex_oi > 4:
        if (NN // tilex_oi) % 2 == 1:
            break
        tilex_oi *= 2

    tiley_ii = 8
    while M % tiley_ii != 0:
        tiley_ii //= 2
    MM = M // tiley_ii
    tiley_oi = 1
    while MM // tiley_oi > 4:
        if (MM // tiley_oi) % 2 == 1:
            break
        tiley_oi *= 2

    cfg["tile_y"] = SplitEntity([MM // tiley_oi, tiley_oi, tiley_ii])
    cfg["tile_x"] = SplitEntity([NN // tilex_oi, tilex_oi, tilex_ii])
    cfg["tile_k"] = SplitEntity([K, 1])
    cfg["tile_inner"] = SplitEntity([M // tiley_ii, tiley_ii])


def _default_dense_nopack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, (tvm.tir.Var, tvm.tir.Any)):
        M = 16
    if isinstance(N, (tvm.tir.Var, tvm.tir.Any)):
        N = 16
    if isinstance(K, (tvm.tir.Var, tvm.tir.Any)):
        K = 16

    vec_width = get_simd_32bit_lanes()
    tilek_bn = 1
    for bn in range(vec_width * 2, 0, -1):
        if K % bn == 0:
            tilek_bn = bn
            break
    cfg["tile_k"] = SplitEntity([K // tilek_bn, tilek_bn])
    cfg["tile_x"] = SplitEntity([N, 1])
    cfg["tile_y"] = SplitEntity([1, M])


@autotvm.register_topi_compute("dense_nopack.x86")
def dense_nopack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense without packing"""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
    if enable_tuning:
        # create tuning space
        cfg.define_split(
            "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=2
        )
        cfg.define_split(
            "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=2
        )
        cfg.define_split(
            "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
        )
    else:
        tile_y_size = [1, 2]
        tile_x_size = [1, 3]
        tile_k_size = [1, 4]
        cfg.define_split(
            "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
            policy="candidate", num_outputs=2, candidate=[tile_y_size]
        )
        cfg.define_split(
            "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N,
            policy="candidate", num_outputs=2, candidate=[tile_x_size]
        )
        cfg.define_split(
            "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K,
            policy="candidate", num_outputs=2, candidate=[tile_k_size]
        )
    if cfg.is_fallback:
        _default_dense_nopack_config(cfg, M, N, K)

    vec = cfg["tile_k"].size[-1]
    k = te.reduce_axis((0, K // vec), "k")
    CC = te.compute(
        (M, N, vec),
        lambda z, y, x: te.sum(
            data[z, k * vec + x].astype(out_dtype) * weight[y, k * vec + x].astype(out_dtype),
            axis=k,
        ),
    )

    kk = te.reduce_axis((0, vec), "kk")
    C = te.compute((M, N), lambda y, x: te.sum(CC[y, x, kk], axis=kk), tag="dense_nopack")
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_schedule("dense_nopack.x86")
def schedule_dense_nopack(cfg, outs):
    """Create the schedule for dense_nopack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_nopack" in op.tag:
            _schedule_dense_nopack_template(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("dense_pack.x86")
def dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense with transformed weight."""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if len(weight.shape) == 3:
        N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
        N = N * packw_bn
    else:
        N, _ = get_const_tuple(weight.shape)  # out_dim
    enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
    enable_tuning = False
    space_name = os.getenv("TVM_TUNING_SPACE_NAME")
    if enable_tuning:
        # create tuning space
        if space_name is None or space_name == "default" or space_name == "large":
            cfg.define_split(
                "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=3
            )
            if space_name == "default":
                cfg.define_split(
                    "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=3
                )
            else:
                cfg.define_split(
                    "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=3,
                    filter=lambda x: x.size[-1] == 4
                )
            cfg.define_split(
                "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
            )
            if space_name == "default":
                cfg.define_split(
                    "tile_inner",
                    32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                    num_outputs=2,
                    filter=lambda y: y.size[-1] <= 16,
                )
            else:
                cfg.define_split(
                    "tile_inner",
                    32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                    num_outputs=2,
                    filter=lambda y: y.size[-1] == 1,
                )
        elif space_name == "small":
            mb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            #kb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            kb_cadidates = [2]
            nb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            mr_cadidates = [4, 8, 16, 32]
            nr_cadidates = [4, 8, 16, 32]
            if DENSE_X86_DEFAULT_CONFIGS["cfg_define_type"] == "knob":
                cfg.define_knob("mb", mb_cadidates)
                cfg.define_knob("kb", kb_cadidates)
                cfg.define_knob("nb", nb_cadidates)
                cfg.define_knob("mr", mr_cadidates)
                cfg.define_knob("nr", nr_cadidates)
            else:
                tile_y_sizes = build_3d_tile_sizes(mb_cadidates, mr_cadidates)
                tile_x_sizes = build_3d_tile_sizes(nb_cadidates, nr_cadidates)
                tile_k_sizes = build_2d_tile_sizes(kb_cadidates)
                cfg.define_split(
                    "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                    policy="candidate", num_outputs=3, candidate=tile_y_sizes
                )
                cfg.define_split(
                    "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N,
                    policy="candidate", num_outputs=3, candidate=tile_x_sizes
                )
                cfg.define_split(
                    "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K,
                    policy="candidate", num_outputs=2, candidate=tile_k_sizes
                )
        else:
            raise ValueError("Unsupported space name: " + space_name)
    else:
        if space_name is None or space_name == "default" or space_name == "large":
            tile_y_size = [1, 40, 16]
            tile_x_size = [1, 12, 16]
            tile_k_size = [1, 2]
            tile_inner_size = [1, 1]
            cfg.define_split(
                "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                policy="candidate", num_outputs=3, candidate=[tile_y_size]
            )
            cfg.define_split(
                "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N,
                policy="candidate", num_outputs=3, candidate=[tile_x_size]
            )
            cfg.define_split(
                "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K,
                policy="candidate", num_outputs=2, candidate=[tile_k_size]
            )
            cfg.define_split(
                "tile_inner",
                32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                policy="candidate",
                num_outputs=2,
                candidate=[tile_inner_size],
            )
        elif space_name == "small":
            if DENSE_X86_DEFAULT_CONFIGS["cfg_define_type"] == "knob":
                cfg.define_knob("mb", [256])
                cfg.define_knob("kb", [16])
                cfg.define_knob("nb", [256])
                cfg.define_knob("mr", [4])
                cfg.define_knob("nr", [4])
            else:
                tile_y_size = [1, 4, 16]
                tile_x_size = [1, 2, 32]
                tile_k_size = [1, 2]
                tile_inner_size = [1, 1]
                cfg.define_split(
                    "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                    policy="candidate", num_outputs=3, candidate=[tile_y_size]
                )
                cfg.define_split(
                    "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N,
                    policy="candidate", num_outputs=3, candidate=[tile_x_size]
                )
                cfg.define_split(
                    "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K,
                    policy="candidate", num_outputs=2, candidate=[tile_k_size]
                )
                cfg.define_split(
                    "tile_inner",
                    32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
                    policy="candidate",
                    num_outputs=2,
                    candidate=[tile_inner_size],
                )
        else:
            raise ValueError("Unsupported space name: " + space_name)
    
    if cfg.is_fallback:
        if DENSE_X86_DEFAULT_CONFIGS["cfg_define_type"] == "knob":
            cfg.define_knob("mb", [32])
            cfg.define_knob("kb", [32])
            cfg.define_knob("nb", [32])
            cfg.define_knob("mr", [4])
            cfg.define_knob("nr", [4])
        else:
            #_default_dense_pack_config(cfg, M, N, K)
            cfg["tile_y"] = SplitEntity([-1, 8, 4])
            cfg["tile_x"] = SplitEntity([-1, 8, 4])
            cfg["tile_k"] = SplitEntity([-1, 4])
            cfg["tile_inner"] = SplitEntity([-1, 1])

    do_pad = False
    
    # Pad input data
    tile_rows_data = cfg["tile_y"].size[-2] * cfg["tile_y"].size[-1]
    tile_cols_data = cfg["tile_k"].size[-1]
    pad_M = 0
    pad_K = 0
    if M % tile_rows_data != 0:
        pad_M = tile_rows_data - (M % tile_rows_data)
    if K % tile_cols_data != 0:
        pad_K = tile_cols_data - (K % tile_cols_data)
    M_padded = M + pad_M
    K_padded = K + pad_K
    pad_before = (0, 0)
    pad_after = (pad_M, pad_K)
    if do_pad and (pad_M != 0 or pad_K != 0):
        data = nn.pad(data, pad_before=pad_before, pad_after=pad_after, name="data_padded")

    # Pad weight
    tile_rows_weight = cfg["tile_x"].size[-2] * cfg["tile_x"].size[-1]
    tile_cols_weight = cfg["tile_k"].size[-1]
    pad_N = 0
    if N % tile_rows_weight != 0:
        pad_N = tile_rows_weight - (N % tile_rows_weight)
    N_padded = N + pad_N
    pad_before = (0, 0)
    pad_after = (pad_N, pad_K)
    if do_pad and (pad_N != 0 or pad_K != 0):
        weight = nn.pad(weight, pad_before=pad_before, pad_after=pad_after, name="weight_padded")

    if len(weight.shape) == 2:
        if space_name == "small" and DENSE_X86_DEFAULT_CONFIGS["cfg_define_type"] == "knob":
            packw_bn = cfg["nr"].val
        else:
            packw_bn = cfg["tile_x"].size[-1]
        packw_shape = (N_padded // packw_bn, K_padded, packw_bn)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            packw = tvm.te.placeholder(packw_shape, weight.dtype, name="packed_weight")
        else:
            packw = te.compute(
                packw_shape, lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight"
            )
    else:
        packw = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K_padded), name="k")
    C = te.compute(
        (M_padded, N_padded),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        name="dense_pack",
    )

    # Unpad output
    if do_pad and (pad_M != 0 or pad_N != 0):
        zero = (
            tvm.tir.const(1, C.dtype) * C[M_padded - 1, N_padded - 1]
            - tvm.tir.const(1, C.dtype) * C[M_padded - 1, N_padded - 1]
        )
        out_shape = (M, N)
        C = te.compute(
            out_shape,
            lambda y, x: (C(y, x) + zero).astype(out_dtype),
            name="dense_pack_output",
        )

    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    #print("dense.py: cfg {}".format(cfg))
    return C


@autotvm.register_topi_schedule("dense_pack.x86")
def schedule_dense_pack(cfg, outs):
    """Create the schedule for dense_pack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        #print("dense.py: op.tag {}".format(op.tag))
        if op.name == "dense_pack":
            _schedule_dense_pack_template(cfg, s, op.output(0), outs[0])
        elif op.name == "dense_pack_output":
            _schedule_dense_pack_template(cfg, s, op.input_tensors[0], outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def dense_vnni_compute(cfg, X, packed_w, bias=None):
    """Compute for uint8 x int8 -> int32 dense"""
    m, k = X.shape
    n_o, _, n_i, _ = packed_w.shape
    ak = te.reduce_axis((0, k), name="k")

    C = te.compute(
        (m, n_o * n_i),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packed_w[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        tag="dense_vnni",
        attrs={"schedule_rule": "meta_schedule.dense_vnni"},
    )

    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j], tag=tag.BROADCAST)

    a_y, _ = C.op.axis
    cfg.define_split("tile_y", a_y, num_outputs=2)

    return C


def dense_vnni_schedule(cfg, s, C, O, do_parallel=True):
    """Schedule dense compute using VNNI vpdpbusd instruction"""
    # C: The output of GEMM
    # O: The output of the fused op
    def split_y(out):
        default_y_split_factor = 32
        a_y = out.op.axis[-2]

        if cfg.is_fallback:
            return s[out].split(a_y, factor=default_y_split_factor)

        return cfg["tile_y"].apply(s, out, a_y)

    (a_k,) = C.op.reduce_axis

    a_yo, a_yi = split_y(C)
    a_xo, a_xi = s[C].split(C.op.axis[-1], factor=16)
    a_ko, a_ki = s[C].split(a_k, factor=4)

    s[C].reorder(a_yo, a_xo, a_yi, a_ko, a_xi, a_ki)

    pc = dot_16x1x16_uint8_int8_int32_cascadelake()
    s[C].tensorize(a_xi, pc)

    if C == O:
        fused = s[O].fuse(a_yo, a_xo)
    else:
        a_yo, a_yi = split_y(O)
        a_xo, a_xi = s[O].split(O.op.axis[-1], factor=16)

        s[O].reorder(a_yo, a_xo, a_yi, a_xi)
        s[O].vectorize(a_xi)
        s[C].compute_at(s[O], a_yi)

        fused = s[O].fuse(a_yo, a_xo)

    if do_parallel:
        s[O].parallel(fused)

    return s, fused


@autotvm.register_topi_compute("dense_vnni.x86")
def dense_vnni(cfg, data, weight, bias=None, out_dtype=None):
    """Compute for uint8 x int8 -> int32 dense"""
    if out_dtype is None:
        out_dtype = data.dtype
    assert len(weight.shape) == 4
    assert data.dtype == "uint8" and weight.dtype == "int8"
    _, _, n_inner, k_inner = get_const_tuple(weight.shape)  # out_dim
    assert n_inner == 16 and k_inner == 4
    return dense_vnni_compute(cfg, data, weight, bias)


@autotvm.register_topi_schedule("dense_vnni.x86")
def schedule_dense_vnni(cfg, outs):
    """Create a schedule for dense_vnni"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_vnni" in op.tag:
            dense_vnni_schedule(cfg, s, op.output(0), outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def matmul_blas_common(cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, lib):
    """Compute matmul/dense using a BLAS library"""
    M, K = get_const_tuple(tensor_a.shape)
    N, _ = get_const_tuple(tensor_b.shape)
    if isinstance(M, int) and isinstance(K, int) and isinstance(N, int):
        cfg.add_flop(M * K * N * 2)
    if tensor_a.dtype == "uint8" and tensor_b.dtype == "int8" and out_dtype == "int32":
        if not hasattr(lib, "matmul_u8s8s32"):
            raise NotImplementedError(
                f"Matmul/Dense with {lib.__name__} for {tensor_a.dtype} is not supported "
                "(matmulu8s8s32 not imlemented)"
            )
        C = lib.matmul_u8s8s32(tensor_a, tensor_b, transpose_a, transpose_b, dtype=out_dtype)
    elif tensor_a.dtype == "float32" or tensor_a.dtype == "float64":
        C = lib.matmul(tensor_a, tensor_b, transpose_a, transpose_b)
    else:
        raise NotImplementedError(
            f"Matmul/Dense with {lib.__name__} for {tensor_a.dtype} is not supported"
        )

    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_compute("dense_cblas.x86")
def dense_cblas(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using cblas. This is an alias of matmul_nt operator."""
    return matmul_blas_common(cfg, data, weight, bias, out_dtype, False, True, cblas)


@autotvm.register_topi_schedule("dense_cblas.x86")
def schedule_dense_cblas(_, outs):
    """Create schedule for dense_cblas. This is an alias of matmul_nt operator."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_mkl.x86")
def dense_mkl(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using mkl. This is an alias of matmul_nt operator."""
    return matmul_blas_common(cfg, data, weight, bias, out_dtype, False, True, mkl)


@autotvm.register_topi_schedule("dense_mkl.x86")
def schedule_dense_mkl(_, outs):
    """Create schedule for dense_mkl. This is an alias of matmul_nt operator."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_dnnl.x86")
def dense_dnnl(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using dnnl. This is an alias of matmul_nt operator."""
    return matmul_blas_common(cfg, data, weight, bias, out_dtype, False, True, dnnl)


@autotvm.register_topi_schedule("dense_dnnl.x86")
def schedule_dense_dnnl(_, outs):
    """Create schedule for dense_dnnl. This is an alias of matmul_nt operator."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("matmul_cblas.x86")
def matmul_cblas(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Compute matmul using cblas."""
    return matmul_blas_common(
        cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, cblas
    )


@autotvm.register_topi_schedule("matmul_cblas.x86")
def schedule_matmul_cblas(_, outs):
    """Create schedule for matmul_cblas."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("matmul_mkl.x86")
def matmul_mkl(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Compute matmul using mkl."""
    return matmul_blas_common(
        cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, mkl
    )


@autotvm.register_topi_schedule("matmul_mkl.x86")
def schedule_matmul_mkl(_, outs):
    """Create schedule for matmul_mkl."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("matmul_dnnl.x86")
def matmul_dnnl(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Compute matmul using dnnl."""
    return matmul_blas_common(
        cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, dnnl
    )


@autotvm.register_topi_schedule("matmul_dnnl.x86")
def schedule_matmul_dnnl(_, outs):
    """Create schedule for matmul_dnnl."""
    return generic.schedule_extern(outs)
