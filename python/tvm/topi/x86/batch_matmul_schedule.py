
from tvm import te
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor
from ..utils import get_os_env_var_bool, get_os_env_var_int, get_os_env_var_int_list


def schedule_batch_matmul_tile(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A, B = op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            enable_cache_write = True
            vec_bits = 128
            enable_unroll = True

            if enable_cache_write:
                CC = s.cache_write(C, "local")

            num_tile_levels = get_os_env_var_int("TVM_TEST_NUM_TILE_LEVEL", 1)

            if num_tile_levels > 0:
                #mr = get_os_env_var_int("TVM_GEMM_MR", 4)
                #kr = get_os_env_var_int("TVM_GEMM_KR", 4)
                #nr = get_os_env_var_int("TVM_GEMM_NR", 4)

                mr_list = get_os_env_var_int_list("TVM_GEMM_MR", [4])
                kr_list = get_os_env_var_int_list("TVM_GEMM_KR", [4])
                nr_list = get_os_env_var_int_list("TVM_GEMM_NR", [4])
                cfg.define_knob("mr", mr_list)
                cfg.define_knob("kr", kr_list)
                cfg.define_knob("nr", nr_list)
                mr = cfg["mr"].val
                kr = cfg["kr"].val
                nr = cfg["nr"].val

                b, y, x = s[O].op.axis
                yo, xo, yi, xi = s[O].tile(y, x, x_factor=mr, y_factor=nr)
                if not enable_cache_write:
                    (k,) = s[O].op.reduce_axis
                    ko, ki = s[O].split(k, kr)
                    s[O].reorder(b, yo, xo, ko, ki, yi, xi)
                
                if num_tile_levels > 1:
                    #mb = get_os_env_var_int("TVM_GEMM_MB", 32)
                    #kb = get_os_env_var_int("TVM_GEMM_KB", 32)
                    #nb = get_os_env_var_int("TVM_GEMM_NB", 32)

                    mb_list = get_os_env_var_int_list("TVM_GEMM_MB", [32])
                    kb_list = get_os_env_var_int_list("TVM_GEMM_KB", [32])
                    nb_list = get_os_env_var_int_list("TVM_GEMM_NB", [32])
                    cfg.define_knob("mb", mb_list)
                    cfg.define_knob("kb", kb_list)
                    cfg.define_knob("nb", nb_list)
                    mb = cfg["mb"].val
                    kb = cfg["kb"].val
                    nb = cfg["nb"].val

                    if mb <= mr or nb <= nr:
                        return
                    if mb % mr > 0 or nb % nr > 0:
                        return

                    yoo, xoo, yoi, xoi = s[O].tile(yo, xo, x_factor=mb // mr, y_factor=nb // nr)
                    if not enable_cache_write:
                        koo, koi = s[O].split(ko, kb // kr)
                        s[O].reorder(b, koo, yoo, xoo, koi, yoi, xoi, ki, yi, xi)
                    else:
                        s[O].reorder(b, yoo, xoo, yoi, xoi, yi, xi)
                        if vec_bits == 128 and nr % 4 == 0:
                            xio, xii = s[O].split(xi, 4)
                            s[O].vectorize(xii)
                            if enable_unroll:
                                s[O].unroll(xio)
                        
                        if enable_unroll:
                            s[O].unroll(yi)

                        s[CC].compute_at(s[O], xoi)
                        (k,) = s[CC].op.reduce_axis
                        ko, ki = s[CC].split(k, kb)
                        _, yi, xi = CC.op.axis
                        s[CC].reorder(ko, ki, yi, xi)

                    if enable_cache_write:
                        if vec_bits == 128 and nr % 4 == 0:
                            xio, xii = s[CC].split(xi, 4)
                            s[CC].vectorize(xii)
                            if enable_unroll:
                                s[CC].unroll(xio)
                        if enable_unroll:
                            s[CC].unroll(yi)

                    if num_tile_levels > 2:
                        assert not enable_cache_write
                        mbb = 128
                        kbb = 128
                        nbb = 128

                        yooo, xooo, yooi, xooi = s[O].tile(yoo, xoo, x_factor=mbb // mb, y_factor=nbb // nb)
                        kooo, kooi = s[O].split(koo, kbb // kb)
                        s[O].reorder(b, kooo, yooo, xooo, kooi, yooi, xooi, koi, yoi, xoi, ki, yi, xi)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_batch_matmul_cache(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A, B = op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            enable_cache_read = get_os_env_var_bool("TVM_TEST_ENABLE_CR", False)
            enable_cache_write = get_os_env_var_bool("TVM_TEST_ENABLE_CW", False)
            vec_bits = get_os_env_var_int("TVM_TEST_VECTORIZE_BITS", 0)
            enable_unroll = get_os_env_var_bool("TVM_TEST_ENABLE_UNROLL", False)

            if enable_cache_read:
                AA = s.cache_read(A, "global", C)
                BB = s.cache_read(B, "global", C)

            if enable_cache_write:
                CC = s.cache_write(C, "local")

            mr = get_os_env_var_int("TVM_GEMM_MR", 4)
            kr = get_os_env_var_int("TVM_GEMM_KR", 4)
            nr = get_os_env_var_int("TVM_GEMM_NR", 4)

            mb = get_os_env_var_int("TVM_GEMM_MB", 32)
            kb = get_os_env_var_int("TVM_GEMM_KB", 32)
            nb = get_os_env_var_int("TVM_GEMM_NB", 32)

            b, y, x = s[O].op.axis
            yo, xo, yi, xi = s[O].tile(y, x, x_factor=mr, y_factor=nr)
            yoo, xoo, yoi, xoi = s[O].tile(yo, xo, x_factor=mb // mr, y_factor=nb // nr)
            s[O].reorder(b, yoo, xoo, yoi, xoi, yi, xi)
            if vec_bits > 0 and nr == 4:
                s[O].vectorize(xi)
            if enable_unroll:
                s[O].unroll(yi)

            if enable_cache_write:
                s[CC].compute_at(s[C], xoi)
                
                (k,) = s[CC].op.reduce_axis
                #ko, ki = s[CC].split(k, kr)
                ko, ki = s[CC].split(k, kb)
                #koo, koi = s[CC].split(ko, kb // kr)

                '''
                b, y, x = CC.op.axis
                yo, xo, yi, xi = s[CC].tile(y, x, x_factor=mr, y_factor=nr)
                yoo, xoo, yoi, xoi = s[CC].tile(yo, xo, x_factor=mb // mr, y_factor=nb // nr)
                if vec_bits > 0 and nr == 4:
                    s[CC].vectorize(xi)
                if enable_unroll:
                    s[CC].unroll(yi)
                s[CC].reorder(b, yoo, xoo, ko, ki, yoi, xoi, yi, xi)
                #s[CC].reorder(b, yoo, xoo, koo, koi, yoi, xoi, ki, yi, xi)
                '''

                _, yi, xi = CC.op.axis
                s[CC].reorder(ko, ki, yi, xi)
                if vec_bits > 0 and nr == 4:
                    s[CC].vectorize(xi)
                if enable_unroll:
                    s[CC].unroll(yi)

                if enable_cache_read:
                    #s[AA].compute_at(s[CC], koo)
                    #s[BB].compute_at(s[CC], koo)
                    #s[AA].compute_at(s[CC], xoi)
                    #s[BB].compute_at(s[CC], xoi)
                    s[AA].compute_at(s[CC], ko)
                    s[BB].compute_at(s[CC], ko)
                    #s[AA].compute_at(s[CC], ki)
                    #s[BB].compute_at(s[CC], ki)
            else:
                (k,) = s[O].op.reduce_axis
                #ko, ki = s[O].split(k, kr)
                ko, ki = s[O].split(k, kb)
                #koo, koi = s[O].split(ko, kb // kr)
                
                #s[O].reorder(b, yoo, xoo, ko, ki, yoi, xoi, yi, xi)
                s[O].reorder(b, yoo, xoo, yoi, xoi, ko, ki, yi, xi)
                #s[O].reorder(b, yoo, xoo, koo, koi, yoi, xoi, ki, yi, xi)
                
                #s[O].pragma(ko, "auto_unroll_max_step", 128)

                if enable_cache_read:
                    #s[AA].compute_at(s[O], koo)
                    #s[BB].compute_at(s[O], koo)
                    #s[AA].compute_at(s[O], xoi)
                    #s[BB].compute_at(s[O], xoi)
                    s[AA].compute_at(s[O], ko)
                    s[BB].compute_at(s[O], ko)
                    #s[AA].compute_at(s[O], ki)
                    #s[BB].compute_at(s[O], ki)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_batch_matmul_vectorize_and_unroll(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A, B = op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            vec_bits = get_os_env_var_int("TVM_TEST_VECTORIZE_BITS", 0)
            enable_unroll = get_os_env_var_bool("TVM_TEST_ENABLE_UNROLL", False)

            mr = get_os_env_var_int("TVM_GEMM_MR", 4)
            kr = get_os_env_var_int("TVM_GEMM_KR", 4)
            nr = get_os_env_var_int("TVM_GEMM_NR", 4)
            if vec_bits > 0:
                nr = vec_bits // 32

            mb = get_os_env_var_int("TVM_GEMM_MB", 32)
            kb = get_os_env_var_int("TVM_GEMM_KB", 32)
            nb = get_os_env_var_int("TVM_GEMM_NB", 32)

            b, y, x = s[O].op.axis
            yo, xo, yi, xi = s[O].tile(y, x, x_factor=mr, y_factor=nr)
            yoo, xoo, yoi, xoi = s[O].tile(yo, xo, x_factor=mb // mr, y_factor=nb // nr)

            (k,) = s[O].op.reduce_axis
            ko, ki = s[O].split(k, kr)
            koo, koi = s[O].split(ko, kb // kr)
            s[O].reorder(b, yoo, xoo, koo, koi, yoi, xoi, ki, yi, xi)

            if vec_bits > 0:
                s[O].vectorize(xi)
            if enable_unroll:
                s[O].unroll(yi)

    traverse_inline(s, outs[0].op, _callback)
    return s
