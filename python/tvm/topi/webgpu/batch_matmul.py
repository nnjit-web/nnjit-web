
import tvm
from tvm import autotvm
from tvm import te
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from .. import nn
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor
from ..utils import get_os_env_var_bool, get_os_env_var_int
from ..utils import build_2d_tile_sizes, build_3d_tile_sizes
from ..utils import build_antares_3d_tile_sizes, build_antares_cpu_4d_tile_sizes, build_antares_gpu_4d_tile_sizes


@autotvm.register_topi_compute("batch_matmul.webgpu")
def batch_matmul(cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True):
    return nn.batch_matmul(
        x,
        y,
        oshape=out_shape,
        out_dtype=out_dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


@autotvm.register_topi_schedule("batch_matmul.webgpu")
def schedule_batch_matmul(cfg, outs):
    import os

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def default_schedule(cfg, op):
        C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, M, N = get_const_tuple(C.shape)
        AA = s.cache_read(A, "shared", [C])
        AL = s.cache_read(AA, "local", [C])
        BB = s.cache_read(B, "shared", [C])
        BL = s.cache_read(BB, "local", [C])
        CC = s.cache_write(C, "local")
        if op not in s.outputs:
            s[C].compute_inline()
            C = s.outputs[0].output(0)

        b, y, x = s[C].op.axis
        (k,) = s[CC].op.reduce_axis

        cfg.define_split("tile_y", y, num_outputs=3)
        cfg.define_split("tile_x", x, num_outputs=3)
        cfg.define_split("tile_k", k, num_outputs=2)
        cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
        target = tvm.target.Target.current()
        if target.kind.name in ["nvptx", "rocm"]:
            # llvm-based backends cannot do non-explicit unrolling
            cfg.define_knob("unroll_explicit", [1])
        else:
            cfg.define_knob("unroll_explicit", [0, 1])

        if cfg.is_fallback:
            y_bn = x_bn = 32
            y_nthreads = x_nthreads = 8
            cfg["tile_x"] = SplitEntity([-1, x_nthreads, x_bn // x_nthreads])
            cfg["tile_y"] = SplitEntity([-1, y_nthreads, y_bn // y_nthreads])
            cfg["tile_k"] = SplitEntity([-1, 8])
            cfg["auto_unroll_max_step"] = OtherOptionEntity(16)
            cfg["unroll_explicit"] = OtherOptionEntity(1)

        by, ty, yi = cfg["tile_y"].apply(s, C, y)
        bx, tx, xi = cfg["tile_x"].apply(s, C, x)

        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")

        s[C].reorder(b, by, bx, ty, tx, yi, xi)
        s[C].bind(b, te.thread_axis("blockIdx.z"))
        s[C].bind(by, te.thread_axis("blockIdx.y"))
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(ty, thread_y)
        s[C].bind(tx, thread_x)
        s[C].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[C].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

        s[CC].compute_at(s[C], tx)
        _, yi, xi = s[CC].op.axis
        ko, ki = cfg["tile_k"].apply(s, CC, k)
        s[CC].reorder(ko, ki, yi, xi)
        s[CC].pragma(ki, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[CC].pragma(ki, "unroll_explicit", cfg["unroll_explicit"].val)

        s[AA].compute_at(s[CC], ko)
        s[AL].compute_at(s[CC], ki)
        s[BB].compute_at(s[CC], ko)
        s[BL].compute_at(s[CC], ki)
        _, y, k = s[AA].op.axis
        ty, yi = s[AA].split(y, nparts=cfg["tile_y"].size[1])
        tx, ki = s[AA].split(k, nparts=cfg["tile_x"].size[1])
        s[AA].reorder(ty, tx, yi, ki)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        s[AA].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[AA].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

        _, x, k = s[BB].op.axis
        ty, xi = s[BB].split(x, nparts=cfg["tile_y"].size[1])
        tx, ki = s[BB].split(k, nparts=cfg["tile_x"].size[1])
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].reorder(ty, tx, xi, ki)
        s[BB].pragma(xi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BB].pragma(xi, "unroll_explicit", cfg["unroll_explicit"].val)

    def plan_threads(op_shape, axes):
        num_step = os.getenv("STEP", "")
        num_step = int(num_step) if num_step else 0
        if not num_step:
            return [1] * len(axes), [1] * len(axes)

        num_threads, init_threads, shape = 256, [1] * len(axes), [4096 if is_base2 else size for size in op_shape]
        for th in range(2, num_threads + 1):
            while num_threads > 1:
                unchanged = True
                for i, x in enumerate(shape):
                    if x % th == 0 and num_threads % th == 0:
                        num_threads //= th
                        shape[i] //= th
                        init_threads[i] *= th
                        unchanged = False
                if unchanged:
                    break
        num_vthreads, init_vthreads = 256, [1] * len(axes)
        for i, x in enumerate(shape):
            if x % 2 == 0 and num_vthreads % 2 == 0:
                num_vthreads //= 2
                shape[i] //= 2
                init_vthreads[i] *= 2
        return init_threads, init_vthreads

    def antares_cpu_web_schedule(cfg, op):
        output = C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, _, K = get_const_tuple(A.shape)
        
        prefix = "MM"
        is_base2 = int(os.environ.get("BASE2", "0")) > 0
        
        init_threads, init_vthreads = plan_threads(output.shape, s[output].op.axis)
        input_tensors = s[output].op.input_tensors

        mb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        kb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        nb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        mr_cadidates = [4, 8, 16, 32]
        nr_cadidates = [4, 8, 16, 32]

        tile_b_sizes = [[-1, 1, 1, 1]]
        tile_y_sizes = build_antares_cpu_4d_tile_sizes(mb_cadidates, mr_cadidates)
        tile_x_sizes = build_antares_cpu_4d_tile_sizes(nb_cadidates, nr_cadidates)
        tile_k_sizes = build_antares_3d_tile_sizes(kb_cadidates)

        data_sizes, reduce_sizes = [], []
        output_shape = get_const_tuple(output.shape)
        cfg.define_split("tile_b", 4096 if is_base2 else output_shape[0], num_outputs=4,
                         policy="candidate", candidate=tile_b_sizes)
        cfg.define_split("tile_y", 4096 if is_base2 else output_shape[1], num_outputs=4,
                         policy="candidate", candidate=tile_y_sizes)
        cfg.define_split("tile_x", 4096 if is_base2 else output_shape[2], num_outputs=4,
                         policy="candidate", candidate=tile_x_sizes)
        cfg.define_knob("vectorize", [0, 1])
        data_sizes.append(cfg["tile_b"].size)
        data_sizes.append(cfg["tile_y"].size)
        data_sizes.append(cfg["tile_x"].size)

        i = 0
        cfg.define_split("tile_k",
                         4096 if is_base2 else K,
                         num_outputs=3,
                         policy="candidate",
                         candidate=tile_k_sizes)
        reduce_sizes.append([-1, 1, cfg["tile_k"].size[1] * cfg["tile_k"].size[2]])

        num_threads, num_vthreads = 1, 1
        for i in range(len(s[output].op.axis)):
            num_threads *= data_sizes[i][2]
            num_vthreads *= data_sizes[i][1] * data_sizes[i][3]

        target = tvm.target.Target.current(allow_none=False)
        assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
        assert num_threads <= target.max_num_threads, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, target.max_num_threads)

        output_local, = s.cache_write([output], "local")

        data_slices = []
        data_slices.append(cfg["tile_b"].apply(s, output, output.op.axis[0]))
        data_slices.append(cfg["tile_y"].apply(s, output, output.op.axis[1]))
        data_slices.append(cfg["tile_x"].apply(s, output, output.op.axis[2]))

        first, second, third, fourth = [x[0] for x in data_slices], \
                                       [x[1] for x in data_slices], \
                                       [x[2] for x in data_slices], \
                                       [x[3] for x in data_slices]

        s[output].reorder(*(first + second + third + fourth))

        s[output_local].compute_at(s[output], third[-1])
        
        s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
        s[output].bind(s[output].fuse(*second), te.thread_axis("threadIdx.x"))

        i = reduce_at = 0
        output_local_rv_o_o, output_local_rv_o_i, output_local_rv_i = cfg["tile_k"].apply(s, output_local, output_local.op.reduce_axis[reduce_at])

        local_slices = []
        local_slices.append(cfg["tile_b"].apply(s, output_local, output_local.op.axis[0]))
        local_slices.append(cfg["tile_y"].apply(s, output_local, output_local.op.axis[1]))
        local_slices.append(cfg["tile_x"].apply(s, output_local, output_local.op.axis[2]))
        first, second, third, fourth = [x[0] for x in local_slices], \
                                       [x[1] for x in local_slices], \
                                       [x[2] for x in local_slices], \
                                       [x[3] for x in local_slices]
        s[output_local].reorder(*(first + second + [output_local_rv_o_o,] + third + [output_local_rv_o_i,] + fourth + [output_local_rv_i]))

        cfg.define_knob("vectorize", [0, 1])
        if cfg["tile_x"].size[-1] % 4 == 0:
            if cfg["tile_x"].size[-1] > 4:
                fourtho, fourthi = s[output_local].split(fourth[-1], 4)
                s[output_local].unroll(fourtho)
                s[output_local].vectorize(fourthi)
            else:
                s[output_local].vectorize(fourth[-1])

        # unroll
        kernel_scope = first[0]
        s[output_local].pragma(kernel_scope, "auto_unroll_max_step", 512)
        s[output_local].pragma(kernel_scope, "unroll_explicit", True)

    def antares_gpu_web_schedule(cfg, op):
        output = C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, _, K = get_const_tuple(A.shape)

        prefix = "MM"
        is_base2 = int(os.environ.get("BASE2", "0")) > 0

        init_threads, init_vthreads = plan_threads(output.shape, s[output].op.axis)
        input_tensors = s[output].op.input_tensors

        mb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        kb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        nb_cadidates = [4, 8, 16, 32, 64, 128, 256]
        mr_cadidates = [4, 8, 16, 32]
        nr_cadidates = [4, 8, 16, 32]

        tile_b_sizes = [[-1, 1, 1, 1]]
        tile_y_sizes = build_antares_gpu_4d_tile_sizes(mb_cadidates, mr_cadidates)
        tile_x_sizes = build_antares_gpu_4d_tile_sizes(nb_cadidates, nr_cadidates)
        tile_k_sizes = build_antares_3d_tile_sizes(kb_cadidates)

        data_sizes, reduce_sizes = [], []
        output_shape = get_const_tuple(output.shape)
        cfg.define_split("tile_b", 4096 if is_base2 else output_shape[0], num_outputs=4,
                         policy="candidate", candidate=tile_b_sizes)
        cfg.define_split("tile_y", 4096 if is_base2 else output_shape[1], num_outputs=4,
                         policy="candidate", candidate=tile_y_sizes)
        cfg.define_split("tile_x", 4096 if is_base2 else output_shape[2], num_outputs=4,
                         policy="candidate", candidate=tile_x_sizes)
        cfg.define_split("tile_k", 4096 if is_base2 else K, num_outputs=3,
                         policy="candidate", candidate=tile_k_sizes)
        cfg.define_knob("vectorize", [0, 1])
        data_sizes.append(cfg["tile_b"].size)
        data_sizes.append(cfg["tile_y"].size)
        data_sizes.append(cfg["tile_x"].size)

        i = 0
        reduce_sizes.append([-1, 1, cfg["tile_k"].size[1] * cfg["tile_k"].size[2]])

        num_threads, num_vthreads = 1, 1
        for i in range(len(s[output].op.axis)):
            num_threads *= data_sizes[i][2]
            num_vthreads *= data_sizes[i][1] * data_sizes[i][3]

        output_local, = s.cache_write([output], "local")

        data_slices = []
        data_slices.append(cfg["tile_b"].apply(s, output, output.op.axis[0]))
        data_slices.append(cfg["tile_y"].apply(s, output, output.op.axis[1]))
        data_slices.append(cfg["tile_x"].apply(s, output, output.op.axis[2]))

        first, second, third, fourth = [x[0] for x in data_slices], [x[1] for x in data_slices], [x[2] for x in data_slices], [x[3] for x in data_slices]
        s[output].reorder(*(first + second + third + fourth))

        s[output_local].compute_at(s[output], third[-1])

        fused_first = s[output].fuse(*first)
        fused_second = s[output].fuse(*second)
        fused_third = s[output].fuse(*third)
        s[output].bind(fused_first, te.thread_axis("blockIdx.x"))
        s[output].bind(fused_second, te.thread_axis("vthread"))
        s[output].bind(fused_third, te.thread_axis("threadIdx.x"))

        i = reduce_at = 0
        output_local_rv_o_o, output_local_rv_o_i, output_local_rv_i = cfg["tile_k"].apply(s, output_local, output_local.op.reduce_axis[reduce_at])

        local_slices = []
        local_slices.append(cfg["tile_b"].apply(s, output_local, output_local.op.axis[0]))
        local_slices.append(cfg["tile_y"].apply(s, output_local, output_local.op.axis[1]))
        local_slices.append(cfg["tile_x"].apply(s, output_local, output_local.op.axis[2]))
        first, second, third, fourth = [x[0] for x in local_slices], \
                                       [x[1] for x in local_slices], \
                                       [x[2] for x in local_slices], \
                                       [x[3] for x in local_slices]
        s[output_local].reorder(*(first + second + [output_local_rv_o_o, output_local_rv_o_i] + third + [output_local_rv_i] + fourth))

        load_stage = []
        for i, load in enumerate(input_tensors):
            load_stage.append(s.cache_read(load, "shared", [output_local]))
            s[load_stage[-1]].compute_at(s[output_local], output_local_rv_o_o)
        for i, load in enumerate(load_stage):
            fused_o = s[load].fuse(*s[load].op.axis)
            val = 1
            fused_o, fused_i = s[load].split(fused_o, factor=val)
            s[load].vectorize(fused_i)
            fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
            s[load].bind(fused_i, te.thread_axis("threadIdx.x"))

        # unroll
        kernel_scope = fourth[0]
        s[output_local].pragma(kernel_scope, "auto_unroll_max_step", 512)
        s[output_local].pragma(kernel_scope, "unroll_explicit", True)

    def _callback(op):
        if "batch_matmul" in op.tag:
            space_name = os.getenv("TVM_TUNING_SPACE_NAME")
            if space_name is None or space_name in ["default", "large"]:
                default_schedule(cfg, op)
            elif space_name == "web":
                #antares_cpu_web_schedule(cfg, op)
                antares_gpu_web_schedule(cfg, op)

    traverse_inline(s, outs[0].op, _callback)
    return s
