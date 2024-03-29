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
"""x86 batch_matmul operators"""
import os
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas, mkl
from .. import generic, nn
from ..transform import layout_transform
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor
from ..utils import get_os_env_var_bool, get_os_env_var_int
from ..utils import build_2d_tile_sizes, build_3d_tile_sizes
from .dense import dense_vnni_schedule
from .injective import schedule_injective_from_existing


@autotvm.register_topi_compute("batch_matmul_vnni.x86")
def batch_matmul_vnni_compute(cfg, x, y, *_):
    """Compute for uint8 x int8 -> int32 batch_matmul"""
    batch, m, k = x.shape
    packed_y_layout = "BNK16n4k"
    packed_y = layout_transform(y, "BNK", packed_y_layout)
    _, n_o, _, n_i, _ = packed_y.shape
    ak = te.reduce_axis((0, k), name="k")

    z = te.compute(
        (batch, m, n_o * n_i),
        lambda b, i, j: te.sum(
            x[b, i, ak].astype("int32")
            * packed_y[b, tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        tag="batch_matmul_vnni",
        attrs={"schedule_rule": "meta_schedule.batch_matmul_vnni"},
    )

    _, a_y, _ = z.op.axis
    cfg.define_split("tile_y", a_y, num_outputs=2)
    cfg.define_knob("layout_trans_compute_root", [0, 1])

    return z


def batch_matmul_vnni_schedule(cfg, s, C, O, layout_trans):
    """Schedule batch_matmul compute using VNNI vpdpbusd instruction"""
    # C: The output of batched GEMM
    # O: The output of the fused op

    # Schedule the GEMM part
    s, fused_inner = dense_vnni_schedule(cfg, s, C, O, do_parallel=False)
    # Parallelize over batch
    fused = s[O].fuse(O.op.axis[0], fused_inner)
    s[O].parallel(fused)

    if cfg["layout_trans_compute_root"].val:
        s[layout_trans].compute_root()
        schedule_injective_from_existing(s, layout_trans)
    else:
        s[layout_trans].compute_at(s[O], fused)
        _, _, _, ni, ki = s[layout_trans].op.axis
        s[layout_trans].vectorize(ki)
        s[layout_trans].unroll(ni)

    return s


@autotvm.register_topi_compute("batch_matmul.x86")
def batch_matmul(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

    Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file.

    tensor_a : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    tensor_b : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    if transpose_a:
        B, K, M = get_const_tuple(tensor_a.shape)
    else:
        B, M, K = get_const_tuple(tensor_a.shape)
    if transpose_b:
        _, N, _ = get_const_tuple(tensor_b.shape)
    else:
        _, _, N = get_const_tuple(tensor_b.shape)

    enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
    space_name = os.getenv("TVM_TUNING_SPACE_NAME")
    if enable_tuning:
        if space_name is None or space_name == "default":
            cfg.define_split("tile_y", M, num_outputs=2)
            cfg.define_split("tile_x", N, num_outputs=2)
            cfg.define_split("tile_k", K, num_outputs=2)
        elif space_name == "large":
            cfg.define_split("tile_y", M, num_outputs=2)
            cfg.define_split("tile_x", N, num_outputs=2, filter=lambda x: x.size[-1] == 4)
            cfg.define_split("tile_k", K, num_outputs=2)
        elif space_name == "small":
            mb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            kb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            nb_cadidates = [4, 8, 16, 32, 64, 128, 256]
            mr_cadidates = [4, 8, 16, 32]
            nr_cadidates = [4, 8, 16, 32]
            tile_y_sizes = build_3d_tile_sizes(mb_cadidates, mr_cadidates)
            tile_x_sizes = build_3d_tile_sizes(nb_cadidates, nr_cadidates)
            tile_k_sizes = build_2d_tile_sizes(kb_cadidates)
            cfg.define_split(
                "tile_y", M,
                policy="candidate", num_outputs=3, candidate=tile_y_sizes
            )
            cfg.define_split(
                "tile_x", N,
                policy="candidate", num_outputs=3, candidate=tile_x_sizes
            )
            cfg.define_split(
                "tile_k", K,
                policy="candidate", num_outputs=2, candidate=tile_k_sizes
            )
        else:
            raise ValueError("Unsupported space name: " + space_name)
    else:
        if space_name is None or space_name == "default":
            cfg.define_split("tile_y", M, num_outputs=2)
            cfg.define_split("tile_x", N, num_outputs=2)
            cfg.define_split("tile_k", K, num_outputs=2)
        elif space_name == "large":
            cfg.define_split("tile_y", M, num_outputs=2)
            cfg.define_split("tile_x", N, num_outputs=2, filter=lambda x: x.size[-1] == 4)
            cfg.define_split("tile_k", K, num_outputs=2)
        elif space_name == "small":
            mb_cadidates = [256]
            kb_cadidates = [256]
            nb_cadidates = [256]
            mr_cadidates = [4]
            nr_cadidates = [8]
            tile_y_sizes = build_3d_tile_sizes(mb_cadidates, mr_cadidates)
            tile_x_sizes = build_3d_tile_sizes(nb_cadidates, nr_cadidates)
            tile_k_sizes = build_2d_tile_sizes(kb_cadidates)
            cfg.define_split(
                "tile_y", M,
                policy="candidate", num_outputs=3, candidate=tile_y_sizes
            )
            cfg.define_split(
                "tile_x", N,
                policy="candidate", num_outputs=3, candidate=tile_x_sizes
            )
            cfg.define_split(
                "tile_k", K,
                policy="candidate", num_outputs=2, candidate=tile_k_sizes
            )
    
    if cfg.is_fallback:
        #_default_batch_matmul_config(cfg, M, N, K)
        cfg["tile_y"] = SplitEntity([-1, 4])
        cfg["tile_x"] = SplitEntity([-1, 4])
        cfg["tile_k"] = SplitEntity([-1, 4])
    
    # Pad
    do_pad = False
    #assert not transpose_a and not transpose_b
    tile_rows_A = cfg["tile_y"].size[-2] * cfg["tile_y"].size[-1]
    tile_cols_A = cfg["tile_k"].size[-1]
    pad_M = 0
    pad_K = 0
    if M % tile_rows_A != 0:
        pad_M = tile_rows_A - (M % tile_rows_A)
    if K % tile_cols_A != 0:
        pad_K = tile_cols_A - (K % tile_cols_A)
    M_padded = M + pad_M
    K_padded = K + pad_K
    pad_before = (0, 0, 0)
    pad_after = (0, pad_M, pad_K)
    if do_pad and (pad_M != 0 or pad_K != 0):
        tensor_a = nn.pad(tensor_a, pad_before=pad_before, pad_after=pad_after, name="A_padded")

    tile_rows_B = cfg["tile_k"].size[-1]
    tile_cols_B = cfg["tile_x"].size[-2] * cfg["tile_x"].size[-1]
    pad_N = 0
    if N % tile_cols_B != 0:
        pad_N = tile_cols_B - (N % tile_cols_B)
    N_padded = N + pad_N
    pad_before = (0, 0, 0)
    pad_after = (0, pad_K, pad_N)
    if do_pad and (pad_K != 0 or pad_N != 0):
        tensor_b = nn.pad(tensor_b, pad_before=pad_before, pad_after=pad_after, name="B_padded")

    if do_pad:
        out_shape = (B, M_padded, N_padded)
    
    output = nn.batch_matmul(
        tensor_a,
        tensor_b,
        out_shape,
        out_dtype,
        transpose_a,
        transpose_b,
    )

    # Unpad
    if do_pad and (pad_M != 0 or pad_N != 0):
        zero = (
            tvm.tir.const(1, output.dtype) * output[B - 1, M_padded - 1, N_padded - 1]
            - tvm.tir.const(1, output.dtype) * output[B - 1, M_padded - 1, N_padded - 1]
        )
        out_shape = (B, M, N)
        output = te.compute(
            out_shape,
            lambda b, y, x: (output(b, y, x) + zero).astype(out_dtype),
            name="batch_matmul_output",
        )

    return output


@autotvm.register_topi_compute("batch_matmul_interleaved.x86")
def batch_matmul_interleaved(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    assert (not transpose_a) and (not transpose_b)
    
    batches, M, K = get_const_tuple(tensor_a.shape)
    _, _, N = get_const_tuple(tensor_b.shape)

    cfg.add_flop(batches * M * N * K * 2)

    M_padded = M
    K_padded = K
    N_padded = N

    space_name = os.getenv("TVM_TUNING_SPACE_NAME")
    if space_name is None or space_name == "default" or space_name == "large":
        cfg.define_split("tile_y", M, num_outputs=3)
        cfg.define_split("tile_x", N, num_outputs=3)
        cfg.define_split("tile_k", K, num_outputs=2)
    else:
        mb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        kb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        nb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        mr_cadidates = [4, 8, 16, 32]
        nr_cadidates = [4, 8, 16, 32]
        tile_y_sizes = build_3d_tile_sizes(mb_cadidates, mr_cadidates)
        tile_x_sizes = build_3d_tile_sizes(nb_cadidates, nr_cadidates)
        tile_k_sizes = build_2d_tile_sizes(kb_cadidates)
        cfg.define_split(
            "tile_y", M,
            policy="candidate", num_outputs=3, candidate=tile_y_sizes
        )
        cfg.define_split(
            "tile_x", N,
            policy="candidate", num_outputs=3, candidate=tile_x_sizes
        )
        cfg.define_split(
            "tile_k", K,
            policy="candidate", num_outputs=2, candidate=tile_k_sizes
        )

    tile_rows_B = cfg["tile_k"].size[-1]
    tile_cols_B = cfg["tile_x"].size[-2] * cfg["tile_x"].size[-1]
    tile_rows_A = cfg["tile_y"].size[-2] * cfg["tile_y"].size[-1]
    tile_cols_A = tile_rows_B

    A_interleaved = te.compute(
        (batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A),
        lambda b, x, y, z, w: tensor_a[b, z + tile_rows_A * x, w + tile_cols_A * y],
        name="A_interleaved",
    )

    B_interleaved = te.compute(
        (batches, K_padded // tile_rows_B, N_padded // tile_cols_B, tile_rows_B, tile_cols_B),
        lambda b, x, y, z, w: tensor_b[b, z + tile_rows_B * x, w + tile_cols_B * y],
        name="weight_block_reshape",
    )

    N_transformed = B_interleaved.shape[2]

    idxm = tvm.tir.indexmod
    k = te.reduce_axis((0, K_padded), "k")

    C_interleaved = te.compute(
        (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_cols_B),
        lambda b, x, y, w, z: te.sum(
            A_interleaved[b, x, k // tile_cols_A, w, idxm(k, tile_cols_A)].astype(out_dtype)
            * B_interleaved[b, k // tile_rows_B, y, idxm(k, tile_rows_B), z].astype(out_dtype),
            axis=k,
        ),
        name="C_interleaved",
    )

    return te.compute(
        (batches, M, N),
        lambda b, x, y: C_interleaved[
            b, x // tile_rows_A, y // tile_cols_B, idxm(x, tile_rows_A), idxm(y, tile_cols_B),
        ].astype(out_dtype),
        name="C",
    )


@autotvm.register_topi_schedule("batch_matmul.x86")
def schedule_batch_matmul(cfg, outs):
    """Schedule for batch_matmul

    Parameters
    ----------
    cfg : ConfigSpace
        AutoTVM tuning space config file.
    outs : Array of Tensor
        The computation graph description of batch_matmul
        in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    from .batch_matmul_schedule import schedule_batch_matmul_tile
    from .batch_matmul_schedule import schedule_batch_matmul_cache
    from .batch_matmul_schedule import schedule_batch_matmul_vectorize_and_unroll
    import os
    schedule_name = os.getenv("TVM_TEST_SCHEDULE_NAME")
    if schedule_name == "tile":
        return schedule_batch_matmul_tile(cfg, outs)
    elif schedule_name == "cache":
        return schedule_batch_matmul_cache(cfg, outs)
    elif schedule_name == "vectorize" or schedule_name == "unroll":
        return schedule_batch_matmul_vectorize_and_unroll(cfg, outs)

    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        #print("batch_matmul.py: op " + str(op))
        def default_schedule(C):
            do_parallel = False
            #C = op.output(0)
            A, B = s[C].op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            CC = s.cache_write(C, "global")

            b, y, x = s[O].op.axis
            yo, yi = cfg["tile_y"].apply(s, O, y)
            xo, xi = cfg["tile_x"].apply(s, O, x)
            s[O].reorder(b, yo, xo, yi, xi)
            bxyo = s[O].fuse(b, yo, xo)
            if do_parallel:
                s[O].parallel(bxyo)

            s[CC].compute_at(s[O], bxyo)
            (k,) = s[CC].op.reduce_axis
            ko, ki = cfg["tile_k"].apply(s, CC, k)

            Crf = s.rfactor(CC, ki)
            s[Crf].compute_at(s[CC], s[CC].op.axis[0])
            _, _, y, x = s[Crf].op.axis
            s[Crf].fuse(y, x)
            s[Crf].vectorize(s[Crf].op.axis[0])
            s[O].pragma(bxyo, "auto_unroll_max_step", 16)

        def my_schedule(C):
            print("batch_matmul.py: use my scheduling")
            #C = op.output(0)
            A, B = s[C].op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            #return

            #CC = s.cache_write(C, "global")

            # create tuning space
            tile_y_size = tile_x_size = tile_k_size = 4
            #tile_y_size = 4; tile_x_size = 8; tile_k_size = 1
            cfg.define_knob("tile_y", [tile_y_size])
            cfg.define_knob("tile_x", [tile_x_size])
            cfg.define_knob("tile_k", [tile_k_size])
            cfg.define_knob("tile_xi", [4])

            b, y, x = s[O].op.axis
            (k,) = s[O].op.reduce_axis

            #return

            if isinstance(cfg["tile_y"], SplitEntity):
                yo, yi = cfg["tile_y"].apply(s, O, y)
                xo, xi = cfg["tile_x"].apply(s, O, x)
                #ko, ki = cfg["tile_k"].apply(s, O, k)
            else:
                yo, yi = s[O].split(y, cfg["tile_y"].val)
                xo, xi = s[O].split(x, cfg["tile_x"].val)
                #ko, ki = s[O].split(k, cfg["tile_k"].val)

                #xo, yo, xi, yi = s[O].tile(y, x, x_factor=cfg["tile_y"].val, y_factor=cfg["tile_y"].val)
            #s[O].reorder(b, yo, xo, yi, xi)
            #s[O].reorder(b, yo, xo, ko, yi, xi, ki)
            #s[O].reorder(b, yo, xo, yi, xi, k)

            #return

            if isinstance(cfg["tile_k"], SplitEntity):
                ko, ki = cfg["tile_k"].apply(s, O, k)
            else:
                ko, ki = s[O].split(k, cfg["tile_k"].val)
            s[O].reorder(b, yo, xo, ko, yi, xi, ki)
            #s[O].reorder(b, yo, xo, yi, xi, ko, ki)
            #s[O].reorder(b, xo, xi, yo, yi, ko, ki)
            #s[O].reorder(b, yo, xo, ko, ki, yi, xi)

            return

            #s[O].split(yo, 4)
            xio, xii = s[O].split(xi, 4)
            #kio, kii = s[O].split(ki, 4)
            #s[O].reorder(b, yo, xo, ko, yi, xio, kio, xii, kii)
            
            #CC0 = op.output(0)
            #CC1 = op.output(1)
            #s[CC1].compute_at(s[CC0], xo)

            #s[O].vectorize(yi)
            #s[O].vectorize(xi)
            #s[O].vectorize(ki)
            s[O].vectorize(xii)

            #s[O].unroll(b)
            s[O].unroll(yi)
            #s[O].unroll(xi)
            #s[O].unroll(ko)
            #s[O].unroll(ki)
            s[O].unroll(xio)

            return

            s[O].pragma(b, "unroll_explicit", 0)
            s[O].pragma(yo, "unroll_explicit", 0)
            s[O].pragma(xo, "unroll_explicit", 0)
            s[O].pragma(ko, "unroll_explicit", 0)
            s[O].pragma(yi, "unroll_explicit", 0)
            s[O].pragma(xi, "unroll_explicit", 0)
            s[O].pragma(ki, "unroll_explicit", 0)
            '''
            bxyo = s[O].fuse(b, yo, xo)
            s[O].parallel(bxyo)

            s[CC].compute_at(s[O], bxyo)
            (k,) = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, cfg["tile_k"].val)

            Crf = s.rfactor(CC, ki)
            s[Crf].compute_at(s[CC], s[CC].op.axis[0])
            _, _, y, x = s[Crf].op.axis
            s[Crf].fuse(y, x)
            s[Crf].vectorize(s[Crf].op.axis[0])
            s[O].pragma(bxyo, "auto_unroll_max_step", 16)
            '''

        def small_set_schedule(C):
            #C = op.output(0)
            A, B = s[C].op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            CC = s.cache_write(C, "local")

            b, y, x = s[O].op.axis
            (k,) = s[CC].op.reduce_axis

            #yo, xo, yi, xi = s[O].tile(y, x, x_factor=cfg["mr"].val, y_factor=cfg["nr"].val)
            #yt, xt, yo, xo = s[O].tile(yo, xo, x_factor=cfg["mb"].val // cfg["mr"].val, y_factor=cfg["nb"].val // cfg["nr"].val)
            #ko, ki = s[CC].split(k, cfg["kb"].val)

            yt, yo, yi = cfg["tile_y"].apply(s, O, y)
            xt, xo, xi = cfg["tile_x"].apply(s, O, x)
            ko, ki = cfg["tile_k"].apply(s, CC, k)
            
            xio, xii = s[O].split(xi, 4)
            s[O].reorder(yt, xt, yo, xo, yi, xio, xii)
            s[O].vectorize(xii)
            s[O].pragma(yi, "auto_unroll_max_step", 256)

            s[CC].compute_at(s[O], xo)

            _, yi, xi = s[CC].op.axis
            xio, xii = s[CC].split(xi, 4)
            s[CC].reorder(ko, ki, yi, xio, xii)
            s[CC].vectorize(xii)
            s[CC].pragma(yi, "auto_unroll_max_step", 256)

        if "batch_matmul" in op.tag:
            C = op.output(0)
        elif op.name == "batch_matmul_output":
            C = op.input_tensors[0]
        else:
            return
        
        space_name = os.getenv("TVM_TUNING_SPACE_NAME")
        if space_name is None or space_name == "default" or cfg.is_fallback:
            default_schedule(C)
            #my_schedule()
        elif space_name == "small":
            small_set_schedule(C)

            #print("batch_matmul.py: cfg {}".format(cfg))
            #print("batch_matmul.py: space_name {}".format(space_name))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_schedule("batch_matmul_interleaved.x86")
def schedule_batch_matmul_interleaved(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        do_cache_write = False
        simd_size = 4
        
        C = op.output(0)
        C_interleaved = C.op.input_tensors[0]
        A_interleaved = C_interleaved.op.input_tensors[0]
        B_interleaved = C_interleaved.op.input_tensors[1]

        mb = A_interleaved.shape[3]
        kb = A_interleaved.shape[4]
        nb = B_interleaved.shape[3]

        mr = cfg["tile_y"].size[-1]
        nr = cfg["tile_x"].size[-1]
        #kr = cfg["tile_k"].size[-1]

        b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]

        if do_cache_write:
            CC = s.cache_write(C_interleaved, "local")

        xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)
        if not do_cache_write:
            k = C_interleaved.op.reduce_axis[0]
            ko, ki = s[C_interleaved].split(k, kb)
            s[C_interleaved].reorder(b, ko, xio, yio, xii, yii, ki)
            s[C_interleaved].unroll(ki)
            #s[C_interleaved].reorder(b, ko, xio, yio, ki, xii, yii)
            #s[C_interleaved].reorder(b, xio, yio, ko, ki, xii, yii)
            #kio, kii = s[C_interleaved].split(ki, 4)
            #s[C_interleaved].reorder(b, ko, xio, yio, kio, xii, yii, kii)
            #s[C_interleaved].unroll(kii)
        else:
            s[C_interleaved].reorder(b, xo, yo, xio, yio, xii, yii)
        yiio, yiii = s[C_interleaved].split(yii, simd_size)
        s[C_interleaved].unroll(xii)
        s[C_interleaved].unroll(yiio)
        s[C_interleaved].vectorize(yiii)
        
        if do_cache_write:
            s[CC].compute_at(s[C_interleaved], yio)

            _, _, _, xii, yii = s[CC].op.axis
            k = CC.op.reduce_axis[0]
            ko, ki = s[CC].split(k, kb)
            #kio, kii = s[CC].split(ki, 4)
            s[CC].reorder(ko, ki, xii, yii)
            #s[CC].reorder(ko, kio, kii, xii, yii)
            yiio, yiii = s[CC].split(yii, simd_size)
            s[CC].unroll(xii)
            s[CC].unroll(yiio)
            #s[CC].unroll(kii)
            s[CC].vectorize(yiii)

        b, m, n = C.op.axis
        _, inner = s[C].split(n, simd_size)
        s[C].vectorize(inner)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_schedule("batch_matmul_vnni.x86")
def schedule_batch_matmul_vnni(cfg, outs):
    """Schedule for batch_matmul_vnni"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul_vnni" in op.tag:
            layout_trans = op.input_tensors[1]
            batch_matmul_vnni_schedule(cfg, s, op.output(0), outs[0], layout_trans)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _default_batch_matmul_config(cfg, M, N, K):
    cfg["tile_k"] = SplitEntity([K // 16, 16])
    x_bn = get_max_power2_factor(N, 8)
    cfg["tile_x"] = SplitEntity([N // x_bn, x_bn])
    y_bn = get_max_power2_factor(M, 8)
    cfg["tile_y"] = SplitEntity([M // y_bn, y_bn])


def batch_matmul_blas_common(cfg, tensor_a, tensor_b, out_shape, trans_a, trans_b, lib):
    """Computes batch matrix multiplication of `tensor_a` and `tensor_b` when `tensor_a` and
    `tensor_b` are data in batch, using one of BLAS libraries. Supports broadcasting in batch
    dimension.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file

    tensor_a : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    tensor_b : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    trans_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    trans_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    lib : A contrib module which implements batch_matmul function
        cblas and mkl are supported

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(tensor_a.shape) == 3 and len(tensor_b.shape) == 3, "only support 3-dim batch_matmul"
    if trans_a:
        XB, XK, M = get_const_tuple(tensor_a.shape)
    else:
        XB, M, XK = get_const_tuple(tensor_a.shape)
    if trans_b:
        YB, N, YK = get_const_tuple(tensor_b.shape)
    else:
        YB, YK, N = get_const_tuple(tensor_a.shape)
    assert (XB == YB) or (YB == 1) or (XB == 1), "batch dimension doesn't match"
    assert XK == YK, "shapes of x and y is inconsistent"
    if out_shape is not None:
        assert out_shape[0] in (XB, YB), "got invalid output shape"
        assert out_shape[1] == M, "got invalid output shape"
        assert out_shape[2] == N, "got invalid output shape"
    cfg.add_flop(XB * M * N * XK * 2)
    return lib.batch_matmul(tensor_a, tensor_b, trans_a, trans_b)


@autotvm.register_topi_compute("batch_matmul_cblas.x86")
def batch_matmul_cblas(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch_matmul using cblas"""
    del out_dtype  # Unused argument
    return batch_matmul_blas_common(
        cfg, tensor_a, tensor_b, out_shape, transpose_a, transpose_b, cblas
    )


@autotvm.register_topi_schedule("batch_matmul_cblas.x86")
def schedule_batch_matmul_cblas(_, outs):
    """Create schedule for batch_matmul_cblas"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("batch_matmul_mkl.x86")
def batch_matmul_mkl(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch_matmul using mkl"""
    del out_dtype  # Unused argument
    return batch_matmul_blas_common(
        cfg, tensor_a, tensor_b, out_shape, transpose_a, transpose_b, mkl
    )


@autotvm.register_topi_schedule("batch_matmul_mkl.x86")
def schedule_batch_matmul_mkl(_, outs):
    """Create schedule for batch_matmul_mul"""
    return generic.schedule_extern(outs)
