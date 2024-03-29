
from tvm import te
from ..utils import traverse_inline, get_const_tuple
from ..utils import get_os_env_var_bool, get_os_env_var_int


def schedule_batch_matmul_tile(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A, B = s[C].op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, N = get_const_tuple(C.shape)

            b, y, x = s[C].op.axis
            (k,) = s[C].op.reduce_axis

            mr = get_os_env_var_int("TVM_GEMM_MR", 4)
            kr = get_os_env_var_int("TVM_GEMM_KR", 4)
            nr = get_os_env_var_int("TVM_GEMM_NR", 4)
            mb = get_os_env_var_int("TVM_GEMM_MB", 32)
            kb = get_os_env_var_int("TVM_GEMM_KB", 32)
            nb = get_os_env_var_int("TVM_GEMM_NB", 32)

            yo, xo, yi, xi = s[C].tile(y, x, x_factor=mr, y_factor=nr)
            by, bx, ty, tx = s[C].tile(yo, xo, x_factor=mb // mr, y_factor=nb // nr)
            ko, ki = s[C].split(k, kb)

            thread_x = te.thread_axis("threadIdx.x")
            thread_y = te.thread_axis("threadIdx.y")

            s[C].reorder(b, by, bx, ty, tx, ko, ki, yi, xi)
            s[C].bind(b, te.thread_axis("blockIdx.z"))
            s[C].bind(by, te.thread_axis("blockIdx.y"))
            s[C].bind(bx, te.thread_axis("blockIdx.x"))
            s[C].bind(ty, thread_y)
            s[C].bind(tx, thread_x)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_batch_matmul_cache_vectorize_unroll(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            enable_cache_read = get_os_env_var_bool("TVM_TEST_ENABLE_CR", False)
            enable_cache_write = get_os_env_var_bool("TVM_TEST_ENABLE_CW", False)
            vec_bits = get_os_env_var_int("TVM_TEST_VECTORIZE_BITS", 0)
            enable_unroll = get_os_env_var_bool("TVM_TEST_ENABLE_UNROLL", False)

            C = op.output(0)
            A, B = s[C].op.input_tensors
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
            _, M, N = get_const_tuple(C.shape)

            if enable_cache_read:
                AA = s.cache_read(A, "shared", [C])
                BB = s.cache_read(B, "shared", [C])
            
            if enable_cache_write:
                CC = s.cache_write(C, "local")

            b, y, x = s[C].op.axis
            
            mr = get_os_env_var_int("TVM_GEMM_MR", 4)
            kr = get_os_env_var_int("TVM_GEMM_KR", 4)
            nr = get_os_env_var_int("TVM_GEMM_NR", 4)
            mb = get_os_env_var_int("TVM_GEMM_MB", 32)
            kb = get_os_env_var_int("TVM_GEMM_KB", 32)
            nb = get_os_env_var_int("TVM_GEMM_NB", 32)

            if vec_bits > 0:
                nr = vec_bits // 32

            '''
            yo, xo, yi, xi = s[C].tile(y, x, x_factor=mr, y_factor=nr)
            by, bx, ty, tx = s[C].tile(yo, xo, x_factor=mb // mr, y_factor=nb // nr)

            thread_x = te.thread_axis("threadIdx.x")
            thread_y = te.thread_axis("threadIdx.y")

            s[C].reorder(b, by, bx, ty, tx, yi, xi)
            s[C].bind(b, te.thread_axis("blockIdx.z"))
            s[C].bind(by, te.thread_axis("blockIdx.y"))
            s[C].bind(bx, te.thread_axis("blockIdx.x"))
            s[C].bind(ty, thread_y)
            s[C].bind(tx, thread_x)

            if vec_bits > 0:
                s[C].vectorize(xi)

            if enable_unroll:
                s[C].unroll(yi)
                if vec_bits == 0:
                    s[C].unroll(xi)
            '''
            
            yo, xo, yi, xi = s[C].tile(y, x, x_factor=mb // mr, y_factor=nb // nr)
            by, bx, ty, tx = s[C].tile(yo, xo, x_factor=mr, y_factor=nr)

            thread_x = te.thread_axis("threadIdx.x")
            thread_y = te.thread_axis("threadIdx.y")
            thread_vx = te.thread_axis("vthread")
            thread_vy = te.thread_axis("vthread")

            '''
            s[C].reorder(b, by, bx, ty, tx, yi, xi)
            s[C].bind(b, te.thread_axis("blockIdx.z"))
            s[C].bind(by, te.thread_axis("blockIdx.y"))
            s[C].bind(bx, te.thread_axis("blockIdx.x"))
            s[C].bind(ty, thread_vy)
            s[C].bind(tx, thread_vx)
            s[C].bind(yi, thread_y)
            s[C].bind(xi, thread_x)

            if enable_unroll:
                s[C].unroll(ty)
                if vec_bits == 0:
                    s[C].unroll(tx)
            '''
            byx = s[C].fuse(by, bx)
            tyx = s[C].fuse(ty, tx)
            yxi = s[C].fuse(yi, xi)

            s[C].bind(byx, te.thread_axis("blockIdx.x"))
            s[C].bind(tyx, thread_vx)
            s[C].bind(yxi, thread_x)

            if enable_cache_write:
                s[CC].compute_at(s[C], tx)
                (k,) = s[CC].op.reduce_axis
                _, yi, xi = s[CC].op.axis
                ko, ki = s[CC].split(k, kb)
                s[CC].reorder(ko, ki, yi, xi)

                if vec_bits > 0:
                    s[CC].vectorize(xi)

                if enable_unroll:
                    s[CC].unroll(yi)

                if enable_cache_read:
                    s[AA].compute_at(s[CC], ko)
                    s[BB].compute_at(s[CC], ko)
                    _, y, k = s[AA].op.axis
                    ty, yi = s[AA].split(y, nparts=mb // mr)
                    tx, ki = s[AA].split(k, nparts=nb // nr)
                    s[AA].reorder(ty, tx, yi, ki)
                    s[AA].bind(ty, thread_y)
                    s[AA].bind(tx, thread_x)
                    _, x, k = s[BB].op.axis
                    ty, xi = s[BB].split(x, nparts=mb // mr)
                    tx, ki = s[BB].split(k, nparts=nb // nr)
                    s[BB].bind(ty, thread_y)
                    s[BB].bind(tx, thread_x)
                    s[BB].reorder(ty, tx, xi, ki)
            else:
                (k,) = s[C].op.reduce_axis
                ko, ki = s[C].split(k, kb)
                #s[C].reorder(b, by, bx, ty, tx, ko, ki, yi, xi)
                s[C].reorder(b, byx, tyx, ko, ki, yxi)

                if enable_cache_read:
                    s[AA].compute_at(s[C], ko)
                    s[BB].compute_at(s[C], ko)
                    _, y, k = s[AA].op.axis
                    ty, yi = s[AA].split(y, nparts=mb // mr)
                    tx, ki = s[AA].split(k, nparts=nb // nr)
                    s[AA].reorder(ty, tx, yi, ki)
                    s[AA].bind(ty, thread_y)
                    s[AA].bind(tx, thread_x)
                    _, x, k = s[BB].op.axis
                    ty, xi = s[BB].split(x, nparts=mb // mr)
                    tx, ki = s[BB].split(k, nparts=nb // nr)
                    s[BB].bind(ty, thread_y)
                    s[BB].bind(tx, thread_x)
                    s[BB].reorder(ty, tx, xi, ki)

    traverse_inline(s, outs[0].op, _callback)
    return s
