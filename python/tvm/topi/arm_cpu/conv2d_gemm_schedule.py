
from ..utils import get_os_env_var_bool, get_os_env_var_int


def schedule_conv2d_gemm_interleaved_tile(cfg, s, out, final_out):
    num_tile_levels = get_os_env_var_int("TVM_TEST_NUM_TILE_LEVEL", 1)

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

    if num_tile_levels > 1:
        mr = get_os_env_var_int("TVM_GEMM_MR", 4)
        kr = get_os_env_var_int("TVM_GEMM_KR", 4)
        nr = get_os_env_var_int("TVM_GEMM_NR", 4)
        xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)
        kio, kii = s[C_interleaved].split(ki, kr)
        s[C_interleaved].reorder(b, xo, yo, ko, kio, xio, yio, kii, xii, yii)
    
        if num_tile_levels > 2:
            mbb = 128
            kbb = 128
            nbb = 128
            xoo, yoo, xoi, yoi = s[C_interleaved].tile(xo, yo, x_factor=mbb // mb, y_factor=nbb // nb)
            koo, koi = s[C_interleaved].split(ko, kbb // kb)
            s[C_interleaved].reorder(b, koo, xoo, yoo, koi, xoi, yoi, kio, xio, yio, kii, xii, yii)

    return s


def schedule_conv2d_gemm_interleaved_cache(cfg, s, out, final_out):
    enable_cache_read = get_os_env_var_bool("TVM_TEST_ENABLE_CR", False)
    enable_cache_write = get_os_env_var_bool("TVM_TEST_ENABLE_CW", False)

    C = None
    C_interleaved = out
    A_interleaved = C_interleaved.op.input_tensors[0]
    B_interleaved = C_interleaved.op.input_tensors[1]

    mb = A_interleaved.shape[3]
    kb = A_interleaved.shape[4]
    nb = B_interleaved.shape[3]

    if enable_cache_read:
        AA = s.cache_read(A_interleaved, "global", C_interleaved)
        BB = s.cache_read(B_interleaved, "global", C_interleaved)

    if enable_cache_write:
        CC = s.cache_write(C_interleaved, "local")

    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]

    mr = get_os_env_var_int("TVM_GEMM_MR", 4)
    kr = get_os_env_var_int("TVM_GEMM_KR", 4)
    nr = get_os_env_var_int("TVM_GEMM_NR", 4)
    xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)

    if enable_cache_write:
        s[CC].compute_at(s[C_interleaved], yio)
        k = CC.op.reduce_axis[0]
        ko, ki = s[CC].split(k, kb)
        kio, kii = s[CC].split(ki, kr)

        b, xo, yo, xi, yi = CC.op.axis[0:5]
        xio, yio, xii, yii = s[CC].tile(xi, yi, x_factor=mr, y_factor=nr)

        s[CC].reorder(b, xo, yo, ko, kio, xio, yio, kii, xii, yii)
        if enable_cache_read:
            s[AA].compute_at(s[CC], ko)
            s[BB].compute_at(s[CC], ko)
    else:
        k = C_interleaved.op.reduce_axis[0]
        ko, ki = s[C_interleaved].split(k, kb)
        kio, kii = s[C_interleaved].split(ki, kr)
        s[C_interleaved].reorder(b, xo, yo, ko, kio, xio, yio, kii, xii, yii)

        if enable_cache_read:
            s[AA].compute_at(s[C_interleaved], ko)
            s[BB].compute_at(s[C_interleaved], ko)


def schedule_conv2d_gemm_interleaved_vectorize_and_unroll(cfg, s, out, final_out):
    vec_bits = get_os_env_var_int("TVM_TEST_VECTORIZE_BITS", 0)
    enable_unroll = get_os_env_var_bool("TVM_TEST_ENABLE_UNROLL", False)

    C = None
    C_interleaved = out
    A_interleaved = C_interleaved.op.input_tensors[0]
    B_interleaved = C_interleaved.op.input_tensors[1]

    mb = A_interleaved.shape[3]
    kb = A_interleaved.shape[4]
    nb = B_interleaved.shape[3]
    
    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]

    mr = get_os_env_var_int("TVM_GEMM_MR", 4)
    kr = get_os_env_var_int("TVM_GEMM_KR", 4)
    nr = get_os_env_var_int("TVM_GEMM_NR", 4)

    if vec_bits > 0:
        num_vec_elems = vec_bits // 32
        nr = num_vec_elems

    xio, yio, xii, yii = s[C_interleaved].tile(xi, yi, x_factor=mr, y_factor=nr)

    k = C_interleaved.op.reduce_axis[0]
    ko, ki = s[C_interleaved].split(k, kb)
    kio, kii = s[C_interleaved].split(ki, kr)
    s[C_interleaved].reorder(b, xo, yo, ko, kio, xio, yio, kii, xii, yii)

    if vec_bits > 0:
        s[C_interleaved].vectorize(yii)

    if enable_unroll:
        s[C_interleaved].unroll(xii)
