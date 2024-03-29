
import tvm
from tvm import te
from tvm.topi import nn
from tvm.autotvm.task.space import AnnotateEntity, ReorderEntity, OtherOptionEntity
from ..utils import get_const_tuple, get_const_int
from ..nn.utils import get_pad_tuple
from ..utils import get_os_env_var_bool, get_os_env_var_int


def configure_knobs(cfg, M, K):
    """Configure auto-tuning knobs for the interleaved strategy"""
    return


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
    #print("tvm.topi.x86.conv2d_gemm.compute_conv2d_gemm_without_weight_transform")
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

    interleave_B = len(B_interleaved_t.shape) != 2
    use_dense_pack = len(B_interleaved_t.shape) == 3

    #  Pad if necessary
    if interleave_B:
        if not use_dense_pack:
            N_transformed = B_interleaved_t.shape[1]
            tile_rows_B = B_interleaved_t.shape[2]  # K
            tile_cols_B = B_interleaved_t.shape[3]  # N
        else:
            #N_transformed = B_interleaved_t.shape[0]
            #tile_cols_B = B_interleaved_t.shape[2]  # N
            N_transformed = B_interleaved_t.shape[1]
            tile_cols_B = B_interleaved_t.shape[3]  # N
            tile_rows_B = cfg["tile_k"].size[-1]  # K
    else:
        N_transformed = 1
        tile_rows_B = cfg["tile_k"].size[-1]  # K
        tile_cols_B = cfg["tile_x"].size[-1] * cfg["tile_x"].size[-2]  # N
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: N_transformed %d, tile_rows_B %d, tile_cols_B %d" % (N_transformed, tile_rows_B, tile_cols_B))

    tile_rows_A = cfg["tile_y"].size[-2] * cfg["tile_y"].size[-1]
    tile_cols_A = tile_rows_B

    pad_M = 0
    pad_K = 0
    pad_N = 0

    if M % tile_rows_A != 0:
        pad_M = tile_rows_A - (M % tile_rows_A)

    if K % tile_cols_A != 0:
        pad_K = tile_cols_A - (K % tile_cols_A)

    if N % tile_cols_B != 0:
        pad_N = tile_cols_B - (N % tile_cols_B)

    M_padded = M + pad_M
    K_padded = K + pad_K
    #N_padded = N_transformed * tile_cols_B
    N_padded = N + pad_N
    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py: tile_rows_A %d, tile_cols_A %d, tile_cols_B %d" % (tile_rows_A, tile_cols_A, tile_cols_B))
        print("conv2d_gemm.py: pad_M %d, pad_K %d, pad_N %d" % (pad_M, pad_K, pad_N))
        print("conv2d_gemm.py: M_padded %d, K_padded %d, N_padded %d" % (M_padded, K_padded, N_padded))

    pad_before = (0, 0, 0)
    pad_after = (0, pad_M, pad_K)

    if pad_M != 0 or pad_K != 0:
        A = nn.pad(A, pad_before=pad_before, pad_after=pad_after, name="A_padded")

    idxm = tvm.tir.indexmod
    k = te.reduce_axis((0, K_padded), "k")

    if interleave_A:
        # Configuration space
        configure_knobs(cfg, M_padded, K_padded)

        if get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
            A_interleaved = te.placeholder(
                (batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A),
                name="A_interleaved"
            )
        else:
            # Pack the input data
            A_interleaved = te.compute(
                (batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A),
                lambda b, x, y, z, w: A[b, z + tile_rows_A * x, w + tile_cols_A * y],
                name="A_interleaved",
            )
        
        # Execute GEMM
        gemm_dtype = out_dtype
        if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
            print("conv2d_gemm.py: b %d, x %d, y %d, w %d, z %d" % (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_cols_B))
            print("conv2d_gemm.py: B_shape %s" % str(B_interleaved_t.shape))
        C_interleaved = te.compute(
            (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_cols_B),
            lambda b, x, y, w, z: te.sum(
                A_interleaved[b, x, k // tile_cols_A, w, idxm(k, tile_cols_A)].astype(gemm_dtype)
                * B_interleaved_t[k // tile_rows_B, y, idxm(k, tile_rows_B), z].astype(gemm_dtype),
                axis=k,
            ),
            name="C_interleaved",
        )

        if get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
            C = C_interleaved
        else:
            # Unpack the result
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
        gemm_dtype = out_dtype
        if interleave_B:
            if not use_dense_pack:
                C = te.compute(
                    (batches, M_padded, N_padded),
                    lambda b, x, y: te.sum(
                        A[b, x, k].astype(gemm_dtype)
                        * B_interleaved_t[
                            k // tile_cols_B, y // tile_rows_B, idxm(k, tile_cols_B), idxm(y, tile_rows_B)
                        ].astype(gemm_dtype),
                        axis=k,
                    ),
                    name="C",
                )
            else:
                packw_bn = cfg["tile_x"].size[-1]
                C = te.compute(
                    (batches, M_padded, N_padded),
                    lambda b, x, y: te.sum(
                        A[b, x, k].astype(gemm_dtype)
                        * B_interleaved_t[
                            y // packw_bn, k, idxm(y, packw_bn)
                        ].astype(gemm_dtype),
                        axis=k,
                    ),
                    name="C",
                )
        else:
            B = B_interleaved_t
            C = te.compute(
                (batches, M_padded, N_padded),
                lambda b, x, y: te.sum(
                    A[b, x, k].astype(gemm_dtype) * B[k, y].astype(gemm_dtype),
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

    if get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
        return C

    # Reshape the result into a convolution output
    out_shape = (batches, OH, OW, OC)
    out = te.compute(
        out_shape,
        lambda b, x, y, z: (C(b, y + OW * x, z) + zero).astype(out_dtype),
        name="conv2d_gemm_output",
    )
    return out


def schedule_conv2d_gemm_interleaved(cfg, s, out, final_out):
    """Schedule the conv2d_gemm interleaved strategy"""
    #print("tvm.topi.x86.conv2d_gemm.schedule_conv2d_gemm_interleaved")

    simd_size = 4
    do_cache_write = False
    
    if get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
        if out.op.name == "C":
            C = out
            C_interleaved = C.op.input_tensors[0]
        else:
            C_interleaved = out
    else:
        C = out.op.input_tensors[0]
        C_interleaved = C.op.input_tensors[0]
    A_interleaved = C_interleaved.op.input_tensors[0]
    B_interleaved = C_interleaved.op.input_tensors[1]

    if not get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
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
            n_outer, n_inner = s[data_im2col].split(n, simd_size)
            s[data_im2col].unroll(n_outer)
            s[data_im2col].vectorize(n_inner)
        else:
            s[data_im2col].compute_inline()

    # Computation
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

    if not get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
        #s[A_interleaved].compute_at(s[C_interleaved], yo)
        #_, _, _, outer_A_interleaved, inner_A_interleaved = A_interleaved.op.axis

        b, m, n = C.op.axis
        _, inner = s[C].split(n, simd_size)
        s[C].vectorize(inner)

        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, simd_size)
        s[out].vectorize(inner)

        # Output transform
        if out != final_out:
            n, h, w, c = out.op.axis
            _, inner = s[out].split(c, simd_size)
            s[C].compute_at(s[out], inner)
            s[out].vectorize(inner)
    
    return s


def schedule_conv2d_gemm_native(cfg, s, out, final_out):
    """Schedule the conv2d_gemm hybrid strategy"""
    #print("tvm.topi.x86.conv2d_gemm.schedule_conv2d_gemm_native")
    simd_size = 4
    if get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
        C = out
    else:
        C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    in_type = A.dtype

    # Computation
    mb = cfg["tile_y"].size[-2]
    kb = cfg["tile_k"].size[-1]
    nb = cfg["tile_x"].size[-2]

    mr = cfg["tile_y"].size[-1]
    nr = cfg["tile_x"].size[-1]
    #kr = cfg["tile_k"].size[-1]

    CC = s.cache_write(C, "local")

    b, y, x = C.op.axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(b, yt, xt, yo, xo, yi, xi)
    s[C].unroll(yi)
    if nr % simd_size == 0:
        xio, xii = s[C].split(xi, simd_size)
        s[C].unroll(xio)
        s[C].vectorize(xii)
    else:
        s[C].vectorize(xi)
    
    s[CC].compute_at(s[C], xo)
    _, yi, xi = CC.op.axis
    (k,) = CC.op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, yi, xi)
    s[CC].unroll(yi)
    if nr % simd_size == 0:
        xio, xii = s[CC].split(xi, simd_size)
        s[CC].unroll(xio)
        s[CC].vectorize(xii)
    else:
        s[CC].vectorize(xi)

    if not get_os_env_var_bool("TVM_DISABLE_CONV2D_GEMM_INPUT_TRANSFORM", False):
        # Input transform
        if A.op.name == "A_padded":
            padding_A = True
            data_im2col = A.op.input_tensors[0]
        else:
            padding_A = False
            data_im2col = A

        b, m, n = data_im2col.op.axis
        if data_im2col.op.name == "data_im2col":
            if data_im2col.shape[2] % simd_size == 0:
                n_outer, n_inner = s[data_im2col].split(n, simd_size)
                #s[data_im2col].unroll(n_outer)
                s[data_im2col].vectorize(n_inner)
        elif padding_A:
            s[data_im2col].compute_inline()
            if nr % simd_size == 0:
                s[A].compute_at(s[CC], xio)
            else:
                s[A].compute_at(s[CC], xi)
        else:
            data_im2col = data_im2col
            #if nr % simd_size == 0:
            #    s[data_im2col].compute_at(s[CC], xio)
            #else:
            #    s[data_im2col].compute_at(s[CC], xi)

        # Output transform
        if out.shape[3] % simd_size == 0:
            n, h, w, c = out.op.axis
            _, inner = s[out].split(c, simd_size)
            s[out].vectorize(inner)

    return s
