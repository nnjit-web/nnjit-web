
import os
from tvm import te, target, autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from ..x86.conv2d import _compute_conv2d_NHWC, _compute_conv2d_NHWC_without_transform
from ..utils import get_const_tuple, traverse_inline
from ..utils import get_os_env_var_bool, get_os_env_var_int


@autotvm.register_topi_compute("conv2d_NHWC_interleaved.webgpu")
def compute_conv2d_NHWC_interleaved(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Interface for interleaved compute_conv2d_NHWC_interleaved"""
    return _compute_conv2d_NHWC(
        cfg, data, kernel, strides, padding, dilation, out_dtype, True
    )


@autotvm.register_topi_compute("conv2d_NHWC_native.webgpu")
def compute_conv2d_NHWC_native(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Interface for native compute_conv2d_NHWC"""
    return _compute_conv2d_NHWC(
        cfg, data, kernel, strides, padding, dilation, out_dtype, False
    )


@autotvm.register_topi_compute("conv2d_NHWC_interleaved_without_transform.webgpu")
def compute_conv2d_NHWC_interleaved_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    """Interface for interleaved compute_conv2d_NHWC_interleaved_without_transform"""
    return _compute_conv2d_NHWC_without_transform(
        cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels, True
    )


def schedule_conv2d_gemm_native_default(cfg, s, conv_out, out):
    print("conv2d_gemm.py: cfg", cfg)
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    B = C.op.input_tensors[1]

    if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
        print("conv2d_gemm.py:", A)
        print("conv2d_gemm.py:", B)
        print("conv2d_gemm.py:", C)

    interleave_B = len(B.shape) == 4

    # Computation
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    CC = s.cache_write(C, "local")

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    if cfg.is_fallback:
        cfg["tile_y"] = SplitEntity([-1, 8, 4])
        cfg["tile_x"] = SplitEntity([-1, 8, 4])
        cfg["tile_k"] = SplitEntity([-1, 32])

    # NOTE(fucheng): If not, extracting tasks from a relay model failed.
    if C.op.name != "C":
        return

    b, y, x = C.op.axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(b, yt, xt, yo, xo, yi, xi)
    #s[C].unroll(yi)
    #s[C].vectorize(xi)
    s[C].bind(b, block_z)
    s[C].bind(yt, block_y)
    s[C].bind(xt, block_x)
    s[C].bind(yo, thread_y)
    s[C].bind(xo, thread_x)
    s[C].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[C].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

    s[CC].compute_at(s[C], xo)
    _, yi, xi = CC.op.axis
    (k,) = CC.op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, yi, xi)
    #s[CC].unroll(yi)
    #s[CC].vectorize(xi)
    s[CC].pragma(ki, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[CC].pragma(ki, "unroll_explicit", cfg["unroll_explicit"].val)

    num_thread_elem_y = cfg["tile_y"].size[-1]
    num_thread_elem_x = cfg["tile_x"].size[-1]
    num_thread_y = cfg["tile_y"].size[-2]
    num_thread_x = cfg["tile_x"].size[-2]

    _, M, N = get_const_tuple(C.shape)
    num_block_y = M // num_thread_y // num_thread_elem_y
    num_block_x = N // num_thread_x // num_thread_elem_x

    #print("num_block_y %d, num_thread_y %d" % (num_block_y, num_thread_y))
    #print("num_block_x %d, num_thread_x %d" % (num_block_x, num_thread_x))

    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    _, y, k = s[AA].op.axis
    ty, yi = s[AA].split(y, nparts=num_thread_y)
    tk, ki = s[AA].split(k, nparts=num_thread_x)
    s[AA].reorder(ty, tk, yi, ki)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tk, thread_x)
    s[AA].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[AA].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

    if interleave_B:
        ko, xt, ki, xo = s[BB].op.axis
        kio, kii = s[BB].split(ki, nparts=num_thread_y)
        xo, xi = s[BB].split(xo, nparts=num_thread_x)
        s[BB].reorder(ko, xt, kio, xo, kii, xi)
        s[BB].bind(kio, thread_y)
        s[BB].bind(xo, thread_x)
        s[BB].pragma(kii, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BB].pragma(kii, "unroll_explicit", cfg["unroll_explicit"].val)
    else:
        k, x = s[BB].op.axis
        tk, ki = s[BB].split(k, nparts=num_thread_y)
        tx, xi = s[BB].split(x, nparts=num_thread_x)
        s[BB].reorder(tk, tx, ki, xi)
        s[BB].bind(tk, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].pragma(ki, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BB].pragma(ki, "unroll_explicit", cfg["unroll_explicit"].val)

    # Input transform
    if A.op.name == "A_padded":
        padding_A = True
        data_im2col = A.op.input_tensors[0]
    else:
        padding_A = False
        data_im2col = A

    b, m, n = data_im2col.op.axis
    if padding_A:
        s[data_im2col].compute_inline()
        #s[A].compute_at(s[CC], xi)
        b, m, n = A.op.axis
        #mo, no, mi, ni = s[A].tile(m, n, x_factor=4, y_factor=4)
        #mt, nt, mo, no = s[A].tile(mo, no, x_factor=8, y_factor=8)
        mo, mi = s[A].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[A].split(mo, nparts=num_block_y)
        no, ni = s[A].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[A].split(no, nparts=num_block_x)
        s[A].reorder(b, mt, nt, mo, no, mi, ni)
        s[A].bind(b, block_z)
        s[A].bind(mt, block_y)
        s[A].bind(nt, block_x)
        s[A].bind(mo, thread_y)
        s[A].bind(no, thread_x)
        s[A].pragma(mi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[A].pragma(mi, "unroll_explicit", cfg["unroll_explicit"].val)
    elif data_im2col.op.name == "data_im2col":
        #n_outer, n_inner = s[data_im2col].split(n, simd_size)
        #mo, no, mi, ni = s[data_im2col].tile(m, n, x_factor=4, y_factor=4)
        #mt, nt, mo, no = s[data_im2col].tile(mo, no, x_factor=8, y_factor=8)
        mo, mi = s[data_im2col].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[data_im2col].split(mo, nparts=num_block_y)
        no, ni = s[data_im2col].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[data_im2col].split(no, nparts=num_block_x)
        s[data_im2col].reorder(b, mt, nt, mo, no, mi, ni)
        s[data_im2col].bind(b, block_z)
        s[data_im2col].bind(mt, block_y)
        s[data_im2col].bind(nt, block_x)
        s[data_im2col].bind(mo, thread_y)
        s[data_im2col].bind(no, thread_x)
        s[data_im2col].pragma(mi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[data_im2col].pragma(mi, "unroll_explicit", cfg["unroll_explicit"].val)
    else:
        #s[data_im2col].compute_at(s[C], xi)
        #mo, no, mi, ni = s[data_im2col].tile(m, n, x_factor=4, y_factor=4)
        #mt, nt, mo, no = s[data_im2col].tile(mo, no, x_factor=8, y_factor=8)
        mo, mi = s[data_im2col].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[data_im2col].split(mo, nparts=num_block_y)
        no, ni = s[data_im2col].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[data_im2col].split(no, nparts=num_block_x)
        s[data_im2col].reorder(b, mt, nt, mo, no, mi, ni)
        s[data_im2col].bind(b, block_z)
        s[data_im2col].bind(mt, block_y)
        s[data_im2col].bind(nt, block_x)
        s[data_im2col].bind(mo, thread_y)
        s[data_im2col].bind(no, thread_x)
        s[data_im2col].pragma(mi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[data_im2col].pragma(mi, "unroll_explicit", cfg["unroll_explicit"].val)

    data = data_im2col.op.input_tensors[0]
    if data.op.name == "data_pad":
        n, h, w, c = data.op.axis
        ho, hi = s[data].split(h, nparts=num_thread_y * num_block_y)
        ht, ho = s[data].split(ho, nparts=num_block_y)
        wo, wi = s[data].split(w, nparts=num_thread_x * num_block_x)
        wt, wo = s[data].split(wo, nparts=num_block_x)
        s[data].reorder(n, ht, wt, ho, wo, hi, wi)
        s[data].bind(n, block_z)
        s[data].bind(ht, block_y)
        s[data].bind(wt, block_x)
        s[data].bind(ho, thread_y)
        s[data].bind(wo, thread_x)
        s[data].pragma(hi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[data].pragma(wi, "unroll_explicit", cfg["unroll_explicit"].val)

    if B.op.name != "weight_transformed":
        if B.op.name == "weight_padding":
            B_flatten = B.op.input_tensors[0]
            s[B_flatten].compute_inline()
        k, x = B.op.axis
        ko, ki = s[B].split(k, nparts=num_thread_y * num_block_y)
        kt, ko = s[B].split(ko, nparts=num_block_y)
        xo, xi = s[B].split(x, nparts=num_thread_x * num_block_x)
        xt, xo = s[B].split(xo, nparts=num_block_x)
        s[B].bind(kt, block_y)
        s[B].bind(xt, block_x)
        s[B].bind(ko, thread_y)
        s[B].bind(xo, thread_x)
        s[B].pragma(ki, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[B].pragma(ki, "unroll_explicit", cfg["unroll_explicit"].val)

    # Output transform
    if out.op.name == "conv2d_gemm_output":
        n, h, w, c = out.op.axis
        #ho, wo, hi, wi = s[out].tile(h, w, x_factor=4, y_factor=4)
        #ht, wt, ho, wo = s[out].tile(ho, wo, x_factor=8, y_factor=8)
        ho, hi = s[out].split(h, nparts=num_thread_y * num_block_y)
        ht, ho = s[out].split(ho, nparts=num_block_y)
        wo, wi = s[out].split(w, nparts=num_thread_x * num_block_x)
        wt, wo = s[out].split(wo, nparts=num_block_x)
        s[out].reorder(n, ht, wt, ho, wo, hi, wi)
        s[out].bind(n, block_z)
        s[out].bind(ht, block_y)
        s[out].bind(wt, block_x)
        s[out].bind(ho, thread_y)
        s[out].bind(wo, thread_x)
        s[out].pragma(hi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[out].pragma(wi, "unroll_explicit", cfg["unroll_explicit"].val)


def schedule_conv2d_gemm_native_small_set(cfg, s, conv_out, out):
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    B = C.op.input_tensors[1]

    interleave_B = len(B.shape) == 4

    # Computation
    simd_size = 4
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    CC = s.cache_write(C, "local")

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    if cfg.is_fallback:
        cfg["tile_y"] = SplitEntity([-1, 8, 4])
        cfg["tile_x"] = SplitEntity([-1, 8, 4])
        cfg["tile_k"] = SplitEntity([-1, 32])
        cfg["vectorize"] = OtherOptionEntity(1)

    # NOTE(fucheng): If not, extracting tasks from a relay model failed.
    if C.op.name != "C":
        return

    b, y, x = C.op.axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(b, yt, xt, yo, xo, yi, xi)
    s[C].unroll(yi)
    if cfg["vectorize"].val:
        xio, xii = s[C].split(xi, simd_size)
        s[C].unroll(xio)
        s[C].vectorize(xii)
    else:
        s[C].unroll(xi)
    s[C].bind(b, block_z)
    s[C].bind(yt, block_y)
    s[C].bind(xt, block_x)
    s[C].bind(yo, thread_y)
    s[C].bind(xo, thread_x)

    s[CC].compute_at(s[C], xo)
    _, yi, xi = CC.op.axis
    (k,) = CC.op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, yi, xi)
    s[CC].unroll(yi)
    if cfg["vectorize"].val:
        xio, xii = s[CC].split(xi, simd_size)
        s[CC].unroll(xio)
        s[CC].vectorize(xii)
    else:
        s[CC].unroll(xi)

    num_thread_elem_y = cfg["tile_y"].size[-1]
    num_thread_elem_x = cfg["tile_x"].size[-1]
    num_thread_y = cfg["tile_y"].size[-2]
    num_thread_x = cfg["tile_x"].size[-2]

    _, M, N = get_const_tuple(C.shape)
    num_block_y = M // num_thread_y // num_thread_elem_y
    num_block_x = N // num_thread_x // num_thread_elem_x

    #print("num_block_y %d, num_thread_y %d" % (num_block_y, num_thread_y))
    #print("num_block_x %d, num_thread_x %d" % (num_block_x, num_thread_x))

    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    _, y, k = s[AA].op.axis
    ty, yi = s[AA].split(y, nparts=num_thread_y)
    tx, ki = s[AA].split(k, nparts=num_thread_x)
    s[AA].reorder(ty, tx, yi, ki)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)

    if interleave_B:
        ko, xt, ki, xo = s[BB].op.axis
        kio, kii = s[BB].split(ki, nparts=num_thread_y)
        xo, xi = s[BB].split(xo, nparts=num_thread_x)
        s[BB].reorder(ko, xt, kio, xo, kii, xi)
        s[BB].bind(kio, thread_y)
        s[BB].bind(xo, thread_x)
    else:
        k, x = s[BB].op.axis
        tk, ki = s[BB].split(k, nparts=num_thread_y)
        tx, xi = s[BB].split(x, nparts=num_thread_x)
        s[BB].reorder(tk, tx, ki, xi)
        s[BB].bind(tk, thread_y)
        s[BB].bind(tx, thread_x)

    # Input transform
    if A.op.name == "A_padded":
        padding_A = True
        data_im2col = A.op.input_tensors[0]
    else:
        padding_A = False
        data_im2col = A

    b, m, n = data_im2col.op.axis
    if padding_A:
        s[data_im2col].compute_inline()
        b, m, n = A.op.axis
        mo, mi = s[A].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[A].split(mo, nparts=num_block_y)
        no, ni = s[A].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[A].split(no, nparts=num_block_x)
        s[A].reorder(b, mt, nt, mo, no, mi, ni)
        s[A].bind(b, block_z)
        s[A].bind(mt, block_y)
        s[A].bind(nt, block_x)
        s[A].bind(mo, thread_y)
        s[A].bind(no, thread_x)
    elif data_im2col.op.name == "data_im2col":
        mo, mi = s[data_im2col].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[data_im2col].split(mo, nparts=num_block_y)
        no, ni = s[data_im2col].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[data_im2col].split(no, nparts=num_block_x)
        s[data_im2col].reorder(b, mt, nt, mo, no, mi, ni)
        s[data_im2col].bind(b, block_z)
        s[data_im2col].bind(mt, block_y)
        s[data_im2col].bind(nt, block_x)
        s[data_im2col].bind(mo, thread_y)
        s[data_im2col].bind(no, thread_x)
    else:
        mo, mi = s[data_im2col].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[data_im2col].split(mo, nparts=num_block_y)
        no, ni = s[data_im2col].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[data_im2col].split(no, nparts=num_block_x)
        s[data_im2col].reorder(b, mt, nt, mo, no, mi, ni)
        s[data_im2col].bind(b, block_z)
        s[data_im2col].bind(mt, block_y)
        s[data_im2col].bind(nt, block_x)
        s[data_im2col].bind(mo, thread_y)
        s[data_im2col].bind(no, thread_x)

    data = data_im2col.op.input_tensors[0]
    if data.op.name == "data_pad":
        n, h, w, c = data.op.axis
        ho, hi = s[data].split(h, nparts=num_thread_y * num_block_y)
        ht, ho = s[data].split(ho, nparts=num_block_y)
        wo, wi = s[data].split(w, nparts=num_thread_x * num_block_x)
        wt, wo = s[data].split(wo, nparts=num_block_x)
        s[data].reorder(n, ht, wt, ho, wo, hi, wi)
        s[data].bind(n, block_z)
        s[data].bind(ht, block_y)
        s[data].bind(wt, block_x)
        s[data].bind(ho, thread_y)
        s[data].bind(wo, thread_x)

    if B.op.name != "weight_transformed":
        if B.op.name == "weight_padding":
            B_flatten = B.op.input_tensors[0]
            s[B_flatten].compute_inline()
        k, x = B.op.axis
        ko, ki = s[B].split(k, nparts=num_thread_y * num_block_y)
        kt, ko = s[B].split(ko, nparts=num_block_y)
        xo, xi = s[B].split(x, nparts=num_thread_x * num_block_x)
        xt, xo = s[B].split(xo, nparts=num_block_x)
        s[B].bind(kt, block_y)
        s[B].bind(xt, block_x)
        s[B].bind(ko, thread_y)
        s[B].bind(xo, thread_x)

    # Output transform
    if out.op.name == "conv2d_gemm_output":
        n, h, w, c = out.op.axis
        ho, hi = s[out].split(h, nparts=num_thread_y * num_block_y)
        ht, ho = s[out].split(ho, nparts=num_block_y)
        wo, wi = s[out].split(w, nparts=num_thread_x * num_block_x)
        wt, wo = s[out].split(wo, nparts=num_block_x)
        s[out].reorder(n, ht, wt, ho, wo, hi, wi)
        s[out].bind(n, block_z)
        s[out].bind(ht, block_y)
        s[out].bind(wt, block_x)
        s[out].bind(ho, thread_y)
        s[out].bind(wo, thread_x)


def schedule_conv2d_gemm_native_small_set_antares(cfg, s, conv_out, out):
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    B = C.op.input_tensors[1]

    interleave_B = len(B.shape) == 4

    # Computation
    simd_size = 4
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    CC = s.cache_write(C, "local")

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    vthread_x = te.thread_axis("vthread")
    vthread_y = te.thread_axis("vthread")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    if cfg.is_fallback:
        cfg["tile_y"] = SplitEntity([-1, 4, 8, 1])
        cfg["tile_x"] = SplitEntity([-1, 4, 8, 1])
        cfg["tile_k"] = SplitEntity([-1, 8, 4])
        cfg["vectorize"] = OtherOptionEntity(1)

    # NOTE(fucheng): If not, extracting tasks from a relay model failed.
    if C.op.name != "C":
        return

    b, y, x = C.op.axis
    y3, y2, y1, y0 = cfg["tile_y"].apply(s, C, y)
    x3, x2, x1, x0 = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(b, y3, x3, y2, x2, y1, x1, y0, x0)
    s[C].unroll(y0)
    if cfg["vectorize"].val:
        x0o, x0i = s[C].split(x0, simd_size)
        s[C].unroll(x0o)
        s[C].vectorize(x0i)
    else:
        s[C].unroll(x0)
    s[C].bind(b, block_z)
    s[C].bind(y3, block_y)
    s[C].bind(x3, block_x)
    s[C].bind(y2, vthread_y)
    s[C].bind(x2, vthread_x)
    s[C].bind(y1, thread_y)
    s[C].bind(x1, thread_x)

    s[CC].compute_at(s[C], x2)

    b, y, x = CC.op.axis
    y3, y2, y1, y0 = cfg["tile_y"].apply(s, CC, y)
    x3, x2, x1, x0 = cfg["tile_x"].apply(s, CC, x)
    s[CC].reorder(b, y3, x3, y2, x2, y1, x1, y0, x0)
    (k,) = CC.op.reduce_axis
    k2, k1, k0 = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(k2, k1, y1, x1, k0, y0, x0)
    s[CC].unroll(y0)
    if cfg["vectorize"].val:
        x0o, x0i = s[CC].split(x0, simd_size)
        s[CC].unroll(x0o)
        s[CC].vectorize(x0i)
    else:
        s[CC].unroll(x0)

    num_thread_elem_y = cfg["tile_y"].size[-1] * cfg["tile_y"].size[-3]
    num_thread_elem_x = cfg["tile_x"].size[-1] * cfg["tile_x"].size[-3]
    num_thread_y = cfg["tile_y"].size[-2]
    num_thread_x = cfg["tile_x"].size[-2]

    _, M, N = get_const_tuple(C.shape)
    num_block_y = M // num_thread_y // num_thread_elem_y
    num_block_x = N // num_thread_x // num_thread_elem_x

    #print("num_block_y %d, num_thread_y %d" % (num_block_y, num_thread_y))
    #print("num_block_x %d, num_thread_x %d" % (num_block_x, num_thread_x))

    ko = k2

    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    _, y, k = s[AA].op.axis
    ty, yi = s[AA].split(y, nparts=num_thread_y)
    tx, ki = s[AA].split(k, nparts=num_thread_x)
    s[AA].reorder(ty, tx, yi, ki)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)

    if interleave_B:
        ko, xt, ki, xo = s[BB].op.axis
        kio, kii = s[BB].split(ki, nparts=num_thread_y)
        xo, xi = s[BB].split(xo, nparts=num_thread_x)
        s[BB].reorder(ko, xt, kio, xo, kii, xi)
        s[BB].bind(kio, thread_y)
        s[BB].bind(xo, thread_x)
    else:
        k, x = s[BB].op.axis
        tk, ki = s[BB].split(k, nparts=num_thread_y)
        tx, xi = s[BB].split(x, nparts=num_thread_x)
        s[BB].reorder(tk, tx, ki, xi)
        s[BB].bind(tk, thread_y)
        s[BB].bind(tx, thread_x)

    # Input transform
    if A.op.name == "A_padded":
        padding_A = True
        data_im2col = A.op.input_tensors[0]
    else:
        padding_A = False
        data_im2col = A

    b, m, n = data_im2col.op.axis
    if padding_A:
        s[data_im2col].compute_inline()
        b, m, n = A.op.axis
        mo, mi = s[A].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[A].split(mo, nparts=num_block_y)
        no, ni = s[A].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[A].split(no, nparts=num_block_x)
        s[A].reorder(b, mt, nt, mo, no, mi, ni)
        s[A].bind(b, block_z)
        s[A].bind(mt, block_y)
        s[A].bind(nt, block_x)
        s[A].bind(mo, thread_y)
        s[A].bind(no, thread_x)
    elif data_im2col.op.name == "data_im2col":
        mo, mi = s[data_im2col].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[data_im2col].split(mo, nparts=num_block_y)
        no, ni = s[data_im2col].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[data_im2col].split(no, nparts=num_block_x)
        s[data_im2col].reorder(b, mt, nt, mo, no, mi, ni)
        s[data_im2col].bind(b, block_z)
        s[data_im2col].bind(mt, block_y)
        s[data_im2col].bind(nt, block_x)
        s[data_im2col].bind(mo, thread_y)
        s[data_im2col].bind(no, thread_x)
    else:
        mo, mi = s[data_im2col].split(m, nparts=num_thread_y * num_block_y)
        mt, mo = s[data_im2col].split(mo, nparts=num_block_y)
        no, ni = s[data_im2col].split(n, nparts=num_thread_x * num_block_x)
        nt, no = s[data_im2col].split(no, nparts=num_block_x)
        s[data_im2col].reorder(b, mt, nt, mo, no, mi, ni)
        s[data_im2col].bind(b, block_z)
        s[data_im2col].bind(mt, block_y)
        s[data_im2col].bind(nt, block_x)
        s[data_im2col].bind(mo, thread_y)
        s[data_im2col].bind(no, thread_x)

    data = data_im2col.op.input_tensors[0]
    if data.op.name == "data_pad":
        n, h, w, c = data.op.axis
        ho, hi = s[data].split(h, nparts=num_thread_y * num_block_y)
        ht, ho = s[data].split(ho, nparts=num_block_y)
        wo, wi = s[data].split(w, nparts=num_thread_x * num_block_x)
        wt, wo = s[data].split(wo, nparts=num_block_x)
        s[data].reorder(n, ht, wt, ho, wo, hi, wi)
        s[data].bind(n, block_z)
        s[data].bind(ht, block_y)
        s[data].bind(wt, block_x)
        s[data].bind(ho, thread_y)
        s[data].bind(wo, thread_x)

    if B.op.name != "weight_transformed":
        if B.op.name == "weight_padding":
            B_flatten = B.op.input_tensors[0]
            s[B_flatten].compute_inline()
        k, x = B.op.axis
        ko, ki = s[B].split(k, nparts=num_thread_y * num_block_y)
        kt, ko = s[B].split(ko, nparts=num_block_y)
        xo, xi = s[B].split(x, nparts=num_thread_x * num_block_x)
        xt, xo = s[B].split(xo, nparts=num_block_x)
        s[B].bind(kt, block_y)
        s[B].bind(xt, block_x)
        s[B].bind(ko, thread_y)
        s[B].bind(xo, thread_x)

    # Output transform
    if out.op.name == "conv2d_gemm_output":
        n, h, w, c = out.op.axis
        ho, hi = s[out].split(h, nparts=num_thread_y * num_block_y)
        ht, ho = s[out].split(ho, nparts=num_block_y)
        wo, wi = s[out].split(w, nparts=num_thread_x * num_block_x)
        wt, wo = s[out].split(wo, nparts=num_block_x)
        s[out].reorder(n, ht, wt, ho, wo, hi, wi)
        s[out].bind(n, block_z)
        s[out].bind(ht, block_y)
        s[out].bind(wt, block_x)
        s[out].bind(ho, thread_y)
        s[out].bind(wo, thread_x)


def _schedule_conv2d_NHWC_native(cfg, outs):
    s = te.create_schedule([x.op for x in outs])
    out = outs[0]

    def _callback(op):
        if op.name == "conv2d_gemm_output":
            conv_out = op.output(0)
            space_name = os.getenv("TVM_TUNING_SPACE_NAME")
            if space_name is None or space_name in ["default", "large"]:
                schedule_conv2d_gemm_native_default(cfg, s, conv_out, out)
            elif space_name == "small":
                #schedule_conv2d_gemm_native_small_set(cfg, s, conv_out, out)
                schedule_conv2d_gemm_native_small_set_antares(cfg, s, conv_out, out)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_conv2d_NHWC(cfg, outs, interleave_A):
    s = te.create_schedule([x.op for x in outs])
    #return s
    
    out = outs[0]

    have_input_transform = get_os_env_var_bool("TVM_CONV2D_GEMM_INPUT_TRANSFORM", True)
    have_output_transform = get_os_env_var_bool("TVM_CONV2D_GEMM_OUTPUT_TRANSFORM", True)

    #x_elem_per_workgroup = get_os_env_var_int("TVM_GEMM_MB", 32)
    #k_elem_per_workgroup = get_os_env_var_int("TVM_GEMM_KB", 32)
    #y_elem_per_workgroup = get_os_env_var_int("TVM_GEMM_NB", 32)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_vx = te.thread_axis("vthread", name="vx")
    thread_vy = te.thread_axis("vthread", name="vy")

    if have_output_transform:
        C = out.op.input_tensors[0]
        C_interleaved = C.op.input_tensors[0]
        s[C].compute_inline()
    else:
        C = None
        C_interleaved = out

    A_interleaved = C_interleaved.op.input_tensors[0]
    B_interleaved = C_interleaved.op.input_tensors[1]

    x_elem_per_workgroup = A_interleaved.shape[3]
    k_elem_per_workgroup = B_interleaved.shape[2]
    y_elem_per_workgroup = B_interleaved.shape[3]

    if x_elem_per_workgroup != y_elem_per_workgroup:
        raise ValueError("x/y elements should have same number")
    
    if have_input_transform:
        A_reshaped = A_interleaved.op.input_tensors[0]
        
        s[A_reshaped].compute_inline()

        b, xo, yo, xi, yi = A_interleaved.op.axis[0:5]
        #s[A_interleaved].bind(xo, te.thread_axis("blockIdx.x"))
        #s[A_interleaved].bind(yo, te.thread_axis("blockIdx.y"))
        #s[A_interleaved].bind(xo, te.thread_axis("threadIdx.x"))
        #s[A_interleaved].bind(yo, te.thread_axis("threadIdx.y"))

    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]

    enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
    if enable_tuning:
        if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
            print("conv2d_gemm.py: x_elem_per_workgroup:", x_elem_per_workgroup)
        x_min_num_threads = 4
        x_min_elem_per_thread = 4
        y_min_num_threads = 4
        y_min_elem_per_thread = 4
        x_num_threads = x_min_num_threads
        x_elem_per_thread = x_elem_per_workgroup // x_num_threads
        x_elem_per_thread_arr = []
        while x_elem_per_thread >= x_min_elem_per_thread:
            if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
                print("conv2d_gemm.py: x_elem_per_thread:", x_elem_per_thread)
            x_elem_per_thread_arr.append(x_elem_per_thread)
            x_num_threads = x_num_threads * 2
            x_elem_per_thread = x_elem_per_workgroup // x_num_threads
        
        if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
            print("conv2d_gemm.py: y_elem_per_workgroup:", y_elem_per_workgroup)
        y_num_threads = y_min_num_threads
        y_elem_per_thread = y_elem_per_workgroup // y_num_threads
        y_elem_per_thread_arr = []
        while y_elem_per_thread >= y_min_elem_per_thread:
            if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
                print("conv2d_gemm.py: y_elem_per_thread:", y_elem_per_thread)
            y_elem_per_thread_arr.append(y_elem_per_thread)
            y_num_threads = y_num_threads * 2
            y_elem_per_thread = y_elem_per_workgroup // y_num_threads

        #x_elem_per_thread_arr = [get_os_env_var_int("TVM_GEMM_MR", 4)]
        #y_elem_per_thread_arr = [get_os_env_var_int("TVM_GEMM_NR", 4)]
        x_elem_per_thread_arr = [4, 8, 16, 32, 64]
        y_elem_per_thread_arr = [4, 8, 16, 32, 64]
        if get_os_env_var_bool("TVM_ENABLE_GEMM_LOG", False):
            print("conv2d_gemm.py: x_elem_per_thread_arr:", x_elem_per_thread_arr)
            print("conv2d_gemm.py: y_elem_per_thread_arr:", y_elem_per_thread_arr)
        #k_elem_per_thread = k_elem_per_workgroup

        if len(x_elem_per_thread_arr) == 0 or len(y_elem_per_thread_arr) == 0:
            raise ValueError("No valid x or y elements per thread")
        
        cfg.define_knob("x_elem_per_thread", x_elem_per_thread_arr)
        cfg.define_knob("y_elem_per_thread", y_elem_per_thread_arr)
        #cfg.define_knob("k_elem_per_thread", [k_elem_per_thread])
        cfg.define_knob("use_cache_read", [True, False])
        cfg.define_knob("use_cache_write", [True])
        cfg.define_knob("use_vthread", [True])
        cfg.define_knob("use_simd", [True, False])

        x_elem_per_thread = cfg["x_elem_per_thread"].val
        y_elem_per_thread = cfg["y_elem_per_thread"].val
        #k_elem_per_thread = cfg["k_elem_per_thread"].val
        k_elem_per_thread = k_elem_per_workgroup

        use_cache_read = cfg["use_cache_read"].val
        use_double_cache_read = False
        use_cache_write = cfg["use_cache_write"].val
        use_vthread = cfg["use_vthread"].val
        use_unroll = True
        use_simd = cfg["use_simd"].val
        simd_size = 4
        use_split_k = B_interleaved.shape[0] > 1
        
        read_cache_type = "shared"
        write_cache_type = "local"

        auto_unroll_max_step = 256
        unroll_explicit = 1
    else:
        use_vthread = False
        use_unroll = True
        use_simd = False
        simd_size = 4
        if not use_vthread:
            x_elem_per_thread = 16
            k_elem_per_thread = k_elem_per_workgroup
            y_elem_per_thread = 16
        else:
            # x/y elem per vthread
            x_elem_per_thread = 8
            k_elem_per_thread = k_elem_per_workgroup
            y_elem_per_thread = 8
            #if use_simd:
            #    y_elem_per_thread = y_elem_per_thread * 4
        
        use_cache_read = True
        use_double_cache_read = False
        use_cache_write = True
        use_split_k = B_interleaved.shape[0] > 1
        
        read_cache_type = "shared"
        if use_double_cache_read:
            read_cache_type = "shared"
        write_cache_type = "local"

        auto_unroll_max_step = 256
        unroll_explicit = 1

    x_num_threads_per_workgroup = x_elem_per_workgroup // x_elem_per_thread
    y_num_threads_per_workgroup = y_elem_per_workgroup // y_elem_per_thread

    if use_cache_read:
        A_interleaved_cached = s.cache_read(A_interleaved, read_cache_type, [C_interleaved])
        B_interleaved_cached = s.cache_read(B_interleaved, read_cache_type, [C_interleaved])
        if use_double_cache_read and read_cache_type == "shared":
            A_interleaved_local_cached = s.cache_read(A_interleaved_cached, "local", [C_interleaved])
            B_interleaved_local_cached = s.cache_read(B_interleaved_cached, "local", [C_interleaved])

    if use_cache_write:
        write_cache_type = "local"
        C_interleaved_cached = s.cache_write(C_interleaved, write_cache_type)

    #if use_cache_read:
        #b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
        #s[A_interleaved_cached].compute_at(s[C_interleaved], yo)

        #b, xo, ko, xi, ki = s[A_interleaved_cached].op.axis
        #xio, xii = s[A_interleaved_cached].split(xi, x_elem_per_thread)
        #s[A_interleaved_cached].bind(xo, block_x)
        #s[A_interleaved_cached].bind(xio, thread_x)

    '''
    b, xo, yo, xi, yi = A_interleaved.op.axis[0:5]
    xio, yio, xii, yii = s[A_interleaved].tile(
        xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
    )
    s[A_interleaved].vectorize(yii)
    s[A_interleaved].unroll(xii)

    s[A_interleaved].bind(xo, block_x)
    s[A_interleaved].bind(yo, block_y)
    s[A_interleaved].bind(xio, thread_x)
    s[A_interleaved].bind(yio, thread_y)
    '''
    
    #if use_cache_write:
        #s[C_interleaved_cached].compute_at(s[C_interleaved], yo)
        #b, xo, yo, xi, yi = s[C_interleaved_cached].op.axis
        #k = C_interleaved_cached.op.reduce_axis[0]
        #C_interleaved = C_interleaved_cached

    #if use_cache_read:
    #    s[A_interleaved_cached].compute_at(s[C_interleaved], xi)

    if use_split_k:
        if not use_cache_write:
            k = C_interleaved.op.reduce_axis[0]
            ko, ki = s[C_interleaved].split(k, B_interleaved.shape[2])
        else:
            k = C_interleaved_cached.op.reduce_axis[0]
            ko, ki = s[C_interleaved_cached].split(k, B_interleaved.shape[2])

    if not use_cache_write:
        b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
        k = C_interleaved.op.reduce_axis[0]
        
        xio, yio, xii, yii = s[C_interleaved].tile(
            xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
        )

        if use_split_k:
            s[C_interleaved].reorder(
                b, xo, yo, xio, yio, ko, ki, xii, yii
            )
        else:
            s[C_interleaved].reorder(
                b, xo, yo, xio, yio, k, xii, yii
            )

        s[C_interleaved].bind(xo, block_x)
        s[C_interleaved].bind(yo, block_y)
        if not use_vthread:
            #s[C_interleaved].unroll(xii)
            if use_simd:
                s[C_interleaved].vectorize(yii)
            #else:
            #    s[C_interleaved].unroll(yii)
            
            s[C_interleaved].bind(xio, thread_x)
            s[C_interleaved].bind(yio, thread_y)
        else:
            s[C_interleaved].bind(xio, thread_vx)
            s[C_interleaved].bind(yio, thread_vy)
            s[C_interleaved].bind(xii, thread_x)
            if not use_simd:
                s[C_interleaved].bind(yii, thread_y)
            else:
                yiio, yiii = s[C_interleaved].split(yii, simd_size)
                s[C_interleaved].bind(yiio, thread_y)
                s[C_interleaved].vectorize(yiii)
    else:
        b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
        if use_vthread:
            s[C_interleaved_cached].compute_at(s[C_interleaved], yi)
        
        '''
        if not use_vthread:
            s[C_interleaved_cached].vectorize(yii)
            s[C_interleaved_cached].unroll(xii)
        else:
            #s[C_interleaved_cached].bind(xio, thread_vx)
            #s[C_interleaved_cached].bind(yio, thread_vy)
            s[C_interleaved_cached].bind(xii, thread_x)
            if not use_simd:
                s[C_interleaved_cached].bind(yii, thread_y)
                s = s
        '''

        b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
        
        s[C_interleaved].bind(xo, block_x)
        s[C_interleaved].bind(yo, block_y)

        b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
        if not use_vthread:
            xio, yio, xii, yii = s[C_interleaved].tile(
                xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
            )
        else:
            if not use_simd:
                xio, yio, xii, yii = s[C_interleaved].tile(
                    xi, yi, x_factor=x_num_threads_per_workgroup, y_factor=y_num_threads_per_workgroup
                )
            else:
                xio, yio, xii, yii = s[C_interleaved].tile(
                    xi, yi, x_factor=x_num_threads_per_workgroup, y_factor=y_num_threads_per_workgroup*4
                )
        s[C_interleaved].reorder(b, xo, yo, xio, yio, xii, yii)

        #s[C_interleaved].compute_at(s[C_interleaved_cached], yio)
        #s[C_interleaved_cached].compute_at(s[C_interleaved], yio)

        #s[C_interleaved].unroll(xii)
        #s[C_interleaved].unroll(yii)
        
        if not use_vthread:
            s[C_interleaved].bind(xio, thread_x)
            s[C_interleaved].bind(yio, thread_y)

            if use_unroll:
                s[C_interleaved].pragma(xii, "auto_unroll_max_step", auto_unroll_max_step)
                s[C_interleaved].pragma(xii, "unroll_explicit", unroll_explicit)

            #s[C_interleaved].unroll(xii)
            if use_simd:
                yiio, yiii = s[C_interleaved].split(yii, simd_size)
                #s[C_interleaved].unroll(yiio)
                s[C_interleaved].vectorize(yiii)

            s[C_interleaved_cached].compute_at(s[C_interleaved], yio)
            
            '''
            b, xo, yo, xi, yi = C_interleaved_cached.op.axis[0:5]
            xio, yio, xii, yii = s[C_interleaved_cached].tile(
                xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
            )
            '''

            _, xo, yo, xi, yi = s[C_interleaved_cached].op.axis
            if use_split_k:
                s[C_interleaved_cached].reorder(
                    ko, ki, xi, yi
                )
            else:
                s[C_interleaved_cached].reorder(
                    k, xi, yi
                )

            #s[C_interleaved_cached].pragma(ki, "auto_unroll_max_step", auto_unroll_max_step)
            #s[C_interleaved_cached].pragma(ki, "unroll_explicit", unroll_explicit)
            xio, yio, xii, yii = s[C_interleaved_cached].tile(
                xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
            )
            if use_unroll:
                s[C_interleaved_cached].pragma(xii, "auto_unroll_max_step", auto_unroll_max_step)
                s[C_interleaved_cached].pragma(xii, "unroll_explicit", unroll_explicit)
            if use_simd:
                yiio, yiii = s[C_interleaved_cached].split(yii, simd_size)
                #s[C_interleaved_cached].unroll(yiio)
                s[C_interleaved_cached].vectorize(yiii)
        else:
            s[C_interleaved].bind(xio, thread_vx)
            s[C_interleaved].bind(yio, thread_vy)
            
            s[C_interleaved].bind(xii, thread_x)
            if not use_simd:
                s[C_interleaved].bind(yii, thread_y)
            else:
                yiio, yiii = s[C_interleaved].split(yii, simd_size)
                s[C_interleaved].bind(yiio, thread_y)
                s[C_interleaved].vectorize(yiii)

                '''
                b, xo, yo, xi, yi = C_interleaved_cached.op.axis[0:5]
                xio, yio, xii, yii = s[C_interleaved_cached].tile(
                    xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
                )
                yiio, yiii = s[C_interleaved_cached].split(yii, simd_size)

                s[C_interleaved_cached].bind(xii, thread_x)
                s[C_interleaved_cached].bind(yiio, thread_y)

                #s[C_interleaved_cached].vectorize(yiii)
                #s[C_interleaved_cached].unroll(xii)
                '''
            
            #s[C_interleaved_cached].pragma(ki, "auto_unroll_max_step", auto_unroll_max_step)
            #s[C_interleaved_cached].pragma(ki, "unroll_explicit", unroll_explicit)

            if use_unroll:
                s[C_interleaved].pragma(xio, "auto_unroll_max_step", auto_unroll_max_step)
                s[C_interleaved].pragma(xio, "unroll_explicit", unroll_explicit)

            _, xo, yo, xi, yi = s[C_interleaved_cached].op.axis
            if use_split_k:
                s[C_interleaved_cached].reorder(
                    ko, ki, xi, yi
                )
            else:
                s[C_interleaved_cached].reorder(
                    k, xi, yi
                )
            #s[C_interleaved_cached].pragma(xi, "auto_unroll_max_step", auto_unroll_max_step)
            #s[C_interleaved_cached].pragma(xi, "unroll_explicit", unroll_explicit)

        #b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
        #s[A_interleaved_cached].compute_at(s[C_interleaved], xi)
        #s[C_interleaved].bind(xo, block_x)
        #s[C_interleaved].bind(yo, block_y)

        #b, xo, yo, xi, yi = C_interleaved_cached.op.axis[0:5]
        #s[C_interleaved_cached].bind(xo, block_x)
        #s[C_interleaved_cached].bind(yo, block_y)
        #s[C_interleaved_cached].bind(xio, thread_x)
        #s[C_interleaved_cached].bind(yio, thread_y)

        #s[out].compute_at(s[C_interleaved_cached], yio)

        '''
        b, xo, yo, xi, yi = out.op.axis[0:5]
        xio, yio, xii, yii = s[out].tile(
            xi, yi, x_factor=x_elem_per_thread, y_factor=y_elem_per_thread
        )

        if write_cache_type == "local":
            s[C_interleaved_cached].compute_at(s[out], yio)
        elif write_cache_type == "shared":
            s[C_interleaved_cached].compute_at(s[out], yo)

        if not use_vthread:
            s[out].vectorize(yii)
            s[out].unroll(xii)

        s[out].bind(xo, block_x)
        s[out].bind(yo, block_y)
        if not use_vthread:
            s[out].bind(xio, thread_x)
            s[out].bind(yio, thread_y)
        else:
            s[out].bind(xio, thread_vx)
            s[out].bind(yio, thread_vy)
            s[out].bind(xii, thread_x)
            if not use_simd:
                s[out].bind(yii, thread_y)
                s = s
            else:
                yiio, yiii = s[out].split(yii, 4)
                s[out].bind(yiio, thread_y)
                s[out].vectorize(yiii)
        '''

        #s[C_interleaved].bind(xo, block_x)
        #s[C_interleaved].bind(yo, block_y)

    if use_cache_read and use_split_k:
        if not use_cache_write:
            s[A_interleaved_cached].compute_at(s[C_interleaved], ko)
            s[B_interleaved_cached].compute_at(s[C_interleaved], ko)
            if use_double_cache_read:
                s[A_interleaved_local_cached].compute_at(s[C_interleaved], ko)
                s[B_interleaved_local_cached].compute_at(s[C_interleaved], ko)
        else:
            s[A_interleaved_cached].compute_at(s[C_interleaved_cached], ko)
            s[B_interleaved_cached].compute_at(s[C_interleaved_cached], ko)
            if use_double_cache_read:
                s[A_interleaved_local_cached].compute_at(s[C_interleaved_cached], ki)
                s[B_interleaved_local_cached].compute_at(s[C_interleaved_cached], ki)
        b, xo, ako, xi, aki = s[A_interleaved_cached].op.axis
        xio, xii = s[A_interleaved_cached].split(xi, x_elem_per_thread)
        akio, akii = s[A_interleaved_cached].split(aki, nparts=y_num_threads_per_workgroup)
        bko, yo, bki, yi = s[B_interleaved_cached].op.axis
        bkio, bkii = s[B_interleaved_cached].split(bki, nparts=x_num_threads_per_workgroup)
        yio, yii = s[B_interleaved_cached].split(yi, y_elem_per_thread)
        if read_cache_type == "shared":
            s[A_interleaved_cached].bind(xio, thread_x)
            #s[A_interleaved_cached].bind(aki, thread_y)
            s[A_interleaved_cached].bind(akio, thread_y)
            #s[A_interleaved_cached].bind(akii, thread_y)
            s[B_interleaved_cached].bind(bkio, thread_x)
            s[B_interleaved_cached].bind(yio, thread_y)
            if use_unroll:
                s[A_interleaved_cached].pragma(xii, "auto_unroll_max_step", auto_unroll_max_step)
                s[A_interleaved_cached].pragma(xii, "unroll_explicit", unroll_explicit)
                s[B_interleaved_cached].pragma(bkii, "auto_unroll_max_step", auto_unroll_max_step)
                s[B_interleaved_cached].pragma(bkii, "unroll_explicit", auto_unroll_max_step)
            #if use_simd:
            #    s[A_interleaved_cached].vectorize(akii)
            #    s[B_interleaved_cached].vectorize(yii)
            '''
            s[A_interleaved_cached].bind(xio, thread_vx)
            s[A_interleaved_cached].bind(akio, thread_vy)
            s[A_interleaved_cached].bind(xii, thread_x)
            s[A_interleaved_cached].bind(akii, thread_y)
            s[B_interleaved_cached].bind(bkio, thread_vx)
            s[B_interleaved_cached].bind(yio, thread_vy)
            s[B_interleaved_cached].bind(bkii, thread_x)
            if not use_simd:
                s[B_interleaved_cached].bind(yii, thread_y)
            else:
                yiio, yiii = s[B_interleaved_cached].split(yii, simd_size)
                s[B_interleaved_cached].bind(yiio, thread_y)
                #s[B_interleaved_cached].vectorize(yiii)
            '''

    #b, xo, yo, xi, xo = A_interleaved_cached.op.axis[0:5]
    #s[A_interleaved_cached].bind(xo, te.thread_axis("blockIdx.x"))
    #s[A_interleaved_cached].bind(yo, te.thread_axis("blockIdx.y"))

    #b, xo, yo = B_interleaved_cached.op.axis[0:5]
    #s[B_interleaved_cached].bind(xo, te.thread_axis("blockIdx.x"))
    #s[B_interleaved_cached].bind(yo, te.thread_axis("blockIdx.y"))

    if have_output_transform:
        b, x, y = out.op.axis[0:3]
        #s[out].bind(x, te.thread_axis("threadIdx.x"))
        #s[out].bind(y, te.thread_axis("threadIdx.y"))

    return s


@autotvm.register_topi_schedule("conv2d_NHWC_native.webgpu")
def schedule_conv2d_NHWC_native(cfg, outs):
    return _schedule_conv2d_NHWC_native(cfg, outs)


@autotvm.register_topi_schedule("conv2d_NHWC_interleaved_without_transform.webgpu")
def schedule_conv2d_NHWC_interleaved_without_transform(cfg, outs):
    """Interface for interleaved schedule_conv2d_NHWC_interleaved"""
    
    from .conv2d_gemm_schedule import _schedule_conv2d_NHWC_tile
    from .conv2d_gemm_schedule import _schedule_conv2d_NHWC_cache_and_vectorize_and_unroll
    import os
    schedule_name = os.getenv("TVM_TEST_SCHEDULE_NAME")
    if schedule_name == "tile":
        return _schedule_conv2d_NHWC_tile(cfg, outs, True)
    elif schedule_name == "cache" or schedule_name == "vectorize" or schedule_name == "unroll":
        return _schedule_conv2d_NHWC_cache_and_vectorize_and_unroll(cfg, outs, True)
    else:
        return _schedule_conv2d_NHWC(cfg, outs, True)
