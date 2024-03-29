
from tvm import te
from ..utils import get_os_env_var_bool, get_os_env_var_int


def _schedule_conv2d_NHWC_tile(cfg, outs, interleave_A):
    num_tile_levels = get_os_env_var_int("TVM_TEST_NUM_TILE_LEVEL", 1)

    s = te.create_schedule([x.op for x in outs])

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    out = outs[0]
    C = None
    C_interleaved = out
    A_interleaved = C_interleaved.op.input_tensors[0]
    B_interleaved = C_interleaved.op.input_tensors[1]

    mb = A_interleaved.shape[3]
    kb = A_interleaved.shape[4]
    nb = B_interleaved.shape[3]

    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
    k = C_interleaved.op.reduce_axis[0]
    ko, ki = s[C_interleaved].split(k, kb)
    s[C_interleaved].reorder(b, xo, yo, ko, ki, xi, yi)

    mr = get_os_env_var_int("TVM_GEMM_MR", 4)
    kr = get_os_env_var_int("TVM_GEMM_KR", 4)
    nr = get_os_env_var_int("TVM_GEMM_NR", 4)
    xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)
    kio, kii = s[C_interleaved].split(ki, kr)
    s[C_interleaved].reorder(b, xo, yo, ko, kio, xio, yio, kii, xii, yii)

    if num_tile_levels == 2:
        s[C_interleaved].bind(xo, block_x)
        s[C_interleaved].bind(yo, block_y)
        s[C_interleaved].bind(xio, thread_x)
        s[C_interleaved].bind(yio, thread_y)

    if num_tile_levels > 2:
        mbb = 128
        kbb = 128
        nbb = 128
        xoo, yoo, xoi, yoi = s[C_interleaved].tile(xo, yo, x_factor=mbb // mb, y_factor=nbb // nb)
        koo, koi = s[C_interleaved].split(ko, kbb // kb)
        s[C_interleaved].reorder(b, koo, xoo, yoo, koi, xoi, yoi, kio, xio, yio, kii, xii, yii)
        s[C_interleaved].bind(xoo, block_x)
        s[C_interleaved].bind(yoo, block_y)
        s[C_interleaved].bind(xio, thread_x)
        s[C_interleaved].bind(yio, thread_y)

    return s


def _schedule_conv2d_NHWC_cache_and_vectorize_and_unroll(cfg, outs, interleave_A):
    enable_cache_read = get_os_env_var_bool("TVM_TEST_ENABLE_CR", False)
    enable_cache_write = get_os_env_var_bool("TVM_TEST_ENABLE_CW", False)
    vec_bits = get_os_env_var_int("TVM_TEST_VECTORIZE_BITS", 0)
    enable_unroll = get_os_env_var_bool("TVM_TEST_ENABLE_UNROLL", False)

    s = te.create_schedule([x.op for x in outs])

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    out = outs[0]
    C = None
    C_interleaved = out
    A_interleaved = C_interleaved.op.input_tensors[0]
    B_interleaved = C_interleaved.op.input_tensors[1]

    if enable_cache_read:
        A_interleaved_cached = s.cache_read(A_interleaved, "shared", [C_interleaved])
        B_interleaved_cached = s.cache_read(B_interleaved, "shared", [C_interleaved])

    if enable_cache_write:
        C_interleaved_cached = s.cache_write(C_interleaved, "local")

    mb = A_interleaved.shape[3]
    kb = A_interleaved.shape[4]
    nb = B_interleaved.shape[3]

    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
    k = C_interleaved.op.reduce_axis[0]

    mr = get_os_env_var_int("TVM_GEMM_MR", 4)
    kr = get_os_env_var_int("TVM_GEMM_KR", 4)
    nr = get_os_env_var_int("TVM_GEMM_NR", 4)
    if vec_bits > 0:
        nr = vec_bits // 32
    xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)

    s[C_interleaved].bind(xo, block_x)
    s[C_interleaved].bind(yo, block_y)
    s[C_interleaved].bind(xio, thread_x)
    s[C_interleaved].bind(yio, thread_y)

    if vec_bits > 0:
        s[C_interleaved].vectorize(yii)
    if enable_unroll:
        s[C_interleaved].unroll(xii)

    if enable_cache_write:
        s[C_interleaved_cached].compute_at(s[C_interleaved], yio)

        ko, ki = s[C_interleaved_cached].split(k, kb)
        kio, kii = s[C_interleaved_cached].split(ki, kr)

        _, xo, yo, xi, yi = s[C_interleaved_cached].op.axis

        if vec_bits > 0:
            xio, yio, xii, yii = s[C_interleaved_cached].tile(xi, yi, x_factor=mr, y_factor=nr)
            s[C_interleaved_cached].reorder(
                ko, kio, xio, yio, kii, xii, yii
            )
            s[C_interleaved_cached].vectorize(yii)
            if enable_unroll:
                s[C_interleaved_cached].unroll(xii)
        else:
            s[C_interleaved_cached].reorder(
                ko, kio, kii, xi, yi
            )

        if enable_cache_read:
            s[A_interleaved_cached].compute_at(s[C_interleaved_cached], ko)
            s[B_interleaved_cached].compute_at(s[C_interleaved_cached], ko)
    else:
        ko, ki = s[C_interleaved].split(k, kb)
        #s[C_interleaved].reorder(b, xo, yo, ko, ki, xio, yio, xii, yii)
        s[C_interleaved].reorder(b, xo, yo, xio, yio, ko, ki, xii, yii)
        #kio, kii = s[C_interleaved].split(ki, kr)
        #s[C_interleaved].reorder(b, xo, yo, ko, kio, xio, yio, kii, xii, yii)

        if vec_bits > 0:
            s[C_interleaved].vectorize(yii)

        if enable_cache_read:
            s[A_interleaved_cached].compute_at(s[C_interleaved], ko)
            s[B_interleaved_cached].compute_at(s[C_interleaved], ko)

    if enable_cache_read:
        b, xo, ako, xi, aki = s[A_interleaved_cached].op.axis
        xio, xii = s[A_interleaved_cached].split(xi, nparts=(mb // mr))
        akio, akii = s[A_interleaved_cached].split(aki, nparts=(nb // nr))
        s[A_interleaved_cached].reorder(b, xo, ako, xio, akio, xii, akii)
        s[A_interleaved_cached].bind(xio, thread_x)
        s[A_interleaved_cached].bind(akio, thread_y)

        bko, yo, bki, yi = s[B_interleaved_cached].op.axis
        bkio, bkii = s[B_interleaved_cached].split(bki, nparts=(mb // mr))
        yio, yii = s[B_interleaved_cached].split(yi, nparts=(nb // nr))
        s[B_interleaved_cached].reorder(bko, yo, bkio, yio, bkii, yii)
        s[B_interleaved_cached].bind(bkio, thread_x)
        s[B_interleaved_cached].bind(yio, thread_y)

    return s
