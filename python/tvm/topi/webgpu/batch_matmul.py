
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
    from .batch_matmul_schedule import schedule_batch_matmul_tile
    from .batch_matmul_schedule import schedule_batch_matmul_cache_vectorize_unroll
    import os
    schedule_name = os.getenv("TVM_TEST_SCHEDULE_NAME")
    if schedule_name == "tile":
        return schedule_batch_matmul_tile(cfg, outs)
    elif schedule_name == "cache" or schedule_name == "vectorize" or schedule_name == "unroll":
        return schedule_batch_matmul_cache_vectorize_unroll(cfg, outs)
    
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def default_schedule(cfg, op):
        #print("batch_matmul.py: default schedule")
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

        enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
        if enable_tuning:
            cfg.define_split("tile_y", y, num_outputs=3)
            cfg.define_split("tile_x", x, num_outputs=3)
            #cfg.define_split("tile_y", y, num_outputs=3,
            #                 filter=lambda y: y.size[-2] >= 4)
            #cfg.define_split("tile_x", x, num_outputs=3,
            #                 filter=lambda x: x.size[-2] >= 4)
            cfg.define_split("tile_k", k, num_outputs=2)
            cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                # llvm-based backends cannot do non-explicit unrolling
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])
        else:
            tile_size = [96, 16, 96, 8, 6]
            #tile_size = [64, 8, 64, 8, 8]
            mb_cadidates = [tile_size[0]]
            kb_cadidates = [tile_size[1]]
            nb_cadidates = [tile_size[2]]
            mr_cadidates = [tile_size[3]]
            nr_cadidates = [tile_size[4]]
            tile_y_sizes = build_3d_tile_sizes(mb_cadidates, mr_cadidates)
            tile_x_sizes = build_3d_tile_sizes(nb_cadidates, nr_cadidates)
            tile_k_sizes = build_2d_tile_sizes(kb_cadidates)
            cfg.define_split("tile_y", y, policy="candidate", num_outputs=3, candidate=tile_y_sizes)
            cfg.define_split("tile_x", x, policy="candidate", num_outputs=3, candidate=tile_x_sizes)
            cfg.define_split("tile_k", k, policy="candidate", num_outputs=2, candidate=tile_k_sizes)
            cfg.define_knob("auto_unroll_max_step", [64])
            cfg.define_knob("unroll_explicit", [1])

        if cfg.is_fallback:
            #y_bn = get_max_power2_factor(M, 64)
            #x_bn = get_max_power2_factor(N, 64)
            #y_nthreads = min(y_bn, 8)
            #x_nthreads = min(x_bn, 8)
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

    def no_local_read_schedule(cfg, op):
        #print("batch_matmul.py: no local read schedule")
        C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, M, N = get_const_tuple(C.shape)
        AA = s.cache_read(A, "shared", [C])
        BB = s.cache_read(B, "shared", [C])
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
            y_bn = get_max_power2_factor(M, 64)
            x_bn = get_max_power2_factor(N, 64)
            y_nthreads = min(y_bn, 8)
            x_nthreads = min(x_bn, 8)
            cfg["tile_x"] = SplitEntity([-1, x_nthreads, x_bn // x_nthreads])
            cfg["tile_y"] = SplitEntity([-1, y_nthreads, y_bn // y_nthreads])
            cfg["tile_k"] = SplitEntity([-1, 8])
            cfg["auto_unroll_max_step"] = OtherOptionEntity(16)

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
        s[BB].compute_at(s[CC], ko)
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

    def no_shared_read_schedule(cfg, op):
        #print("batch_matmul.py: no shared read schedule")
        C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, M, N = get_const_tuple(C.shape)
        CC = s.cache_write(C, "local")
        if op not in s.outputs:
            s[C].compute_inline()
            C = s.outputs[0].output(0)

        b, y, x = s[C].op.axis
        (k,) = s[CC].op.reduce_axis

        enable_tuning = False
        if enable_tuning:
            #cfg.define_split("tile_y", y, num_outputs=3)
            #cfg.define_split("tile_x", x, num_outputs=3)
            cfg.define_split("tile_y", y, num_outputs=3, filter=lambda y: y.size[-2] <= 32 and y.size[-1] <= 4)
            cfg.define_split("tile_x", x, num_outputs=3, filter=lambda x: x.size[-2] <= 32 and x.size[-1] <= 4)
            cfg.define_split("tile_k", k, num_outputs=2)
            cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                # llvm-based backends cannot do non-explicit unrolling
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])
        else:
            cfg.define_split("tile_y", y, policy="candidate", num_outputs=3, candidate=[[1, 16, 4]])
            cfg.define_split("tile_x", x, policy="candidate", num_outputs=3, candidate=[[1, 16, 4]])
            cfg.define_split("tile_k", k, policy="candidate", num_outputs=2, candidate=[[1, 2]])
            cfg.define_knob("auto_unroll_max_step", [64])
            cfg.define_knob("unroll_explicit", [1])

        if cfg.is_fallback:
            y_bn = get_max_power2_factor(M, 64)
            x_bn = get_max_power2_factor(N, 64)
            y_nthreads = min(y_bn, 8)
            x_nthreads = min(x_bn, 8)
            cfg["tile_x"] = SplitEntity([-1, x_nthreads, x_bn // x_nthreads])
            cfg["tile_y"] = SplitEntity([-1, y_nthreads, y_bn // y_nthreads])
            cfg["tile_k"] = SplitEntity([-1, 8])
            cfg["auto_unroll_max_step"] = OtherOptionEntity(16)

        by, ty, yi = cfg["tile_y"].apply(s, C, y)
        bx, tx, xi = cfg["tile_x"].apply(s, C, x)

        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")

        s[C].reorder(b, by, bx, ty, tx, yi, xi)
        #s[C].bind(b, te.thread_axis("blockIdx.z"))
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

    def mcpu_schedule(cfg, op):
        prefix = "MM"
        output = op.output(0)

        hyper_params = [[-1, 2, 8, 4], [-1, 1, 512, 1]]
        
        slice_data, slice_reduce = [], []
        for i in range(len(output.op.axis)):
            slice_data.append(cfg.define_split(
                f"{prefix}:D{i}",
                output.op.axis[i],
                num_outputs=4,
                init_vals=[hyper_params[i % len(hyper_params)],]
            ))
        for i in range(len(output.op.reduce_axis)):
            slice_reduce.append(cfg.define_split(
                f"{prefix}:R{i}",
                output.op.reduce_axis[i],
                num_outputs=2,
                init_vals=[[-1, 4],]
            ))

        unroll = cfg.define_knob(f"{prefix}:UN",
                                 [1, 4, 8, 16, 32, 64],)

        output_local, = s.cache_write([output], "local")

        slice_axes = []
        for i in range(len(output.op.axis)):
            slice_axes.append(cfg[f"{prefix}:D{i}"].apply(
                s,
                output_local,
                output_local.op.axis[i]
            ))

        if output.op.reduce_axis:
            reduce_at = 0
            output_local_K_o, output_local_K_i = cfg[f"{prefix}:R{reduce_at}"].apply(
                s,
                output_local,
                output_local.op.reduce_axis[reduce_at]
            )
            output_local_K_o, output_local_K_i = [output_local_K_o], [output_local_K_i]
        else:
            output_local_K_o, output_local_K_i = [], []

        first, second, third, fourth = [x[0] for x in slice_axes], \
                                       [x[1] for x in slice_axes], \
                                       [x[2] for x in slice_axes], \
                                       [x[3] for x in slice_axes]
        s[output_local].reorder(*(first + second + output_local_K_o + third + output_local_K_i + fourth))

        slice_global_axes = []
        for i in range(len(output.op.axis)):
            if cfg.define_knob(f"{prefix}:_{i}", [False, True]):
                slice_global_axes.append(cfg[f"{prefix}:D{i}"].apply(
                    s,
                    output,
                    output.op.axis[i]
                ))
            else:
                slice_global_axes.append(cfg[f"{prefix}:D{i}"].apply(
                    s,
                    output,
                    output.op.axis[i]
                ))

        s[output].reorder(*([x[0] for x in slice_global_axes] + \
                            [x[1] for x in slice_global_axes] + \
                            [x[2] for x in slice_global_axes]))

        s[output_local].compute_at(s[output], slice_global_axes[-1][1])
        s[output].bind(s[output].fuse(*[x[0] for x in slice_global_axes]), te.thread_axis('threadIdx.x'))

        s[output_local].pragma(first[0], "auto_unroll_max_step", cfg[f"{prefix}:UN"].val)
        s[output_local].pragma(first[0], "unroll_explicit", True)
        #s[output_local].vectorize(fourth[-1])
        s[output_local].unroll(fourth[-1])

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
    
    def antares_cpu_schedule(cfg, op):
        output = C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, _, K = get_const_tuple(A.shape)
        
        prefix = "MM"
        is_base2 = int(os.environ.get("BASE2", "0")) > 0
        
        init_threads, init_vthreads = plan_threads(output.shape, s[output].op.axis)
        input_tensors = s[output].op.input_tensors

        data_sizes, reduce_sizes = [], []
        for i, ax in enumerate(s[output].op.axis):
            output_shape = get_const_tuple(output.shape)
            #cfg.define_split(f"{prefix}D{i}", 4096 if is_base2 else output_shape[i], num_outputs=4)
            #cfg.define_split(f"{prefix}D{i}", 4096 if is_base2 else output_shape[i], num_outputs=4, filter=lambda x: x.size[-1] == 1)
            # Generate tile sizes.
            if i > 0:
                candidate_sizes = [2, 4, 8, 16]
                candidate_tile_sizes = []
                for ii in candidate_sizes:
                    for jj in candidate_sizes:
                        candidate_tile_sizes.append([-1, ii, jj, 1])
            else:
                candidate_tile_sizes = [[-1, 1, 1, 1]]
            cfg.define_split(f"{prefix}D{i}", 4096 if is_base2 else output_shape[i], num_outputs=4, policy="candidate", candidate=candidate_tile_sizes)
            data_sizes.append(cfg[f"{prefix}D{i}"].size)
        for i, ax in enumerate(s[output].op.reduce_axis):
            #cfg.define_split(f"{prefix}R{i}", 4096 if is_base2 else K, num_outputs=3)
            # Generate tile sizes.
            candidate_sizes = [2, 4, 8]
            candidate_tile_sizes = []
            for ii in candidate_sizes:
                for jj in candidate_sizes:
                    candidate_tile_sizes.append([-1, ii, jj])
            cfg.define_split(f"{prefix}R{i}", 4096 if is_base2 else K, num_outputs=3, policy="candidate", candidate=candidate_tile_sizes)
            reduce_sizes.append([-1, 1, cfg[f"{prefix}R{i}"].size[1] * cfg[f"{prefix}R{i}"].size[2]])

        num_threads, num_vthreads = 1, 1
        for i in range(len(s[output].op.axis)):
            num_threads *= data_sizes[i][2]
            num_vthreads *= data_sizes[i][1] * data_sizes[i][3]

        target = tvm.target.Target.current(allow_none=False)
        assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
        assert num_threads <= target.max_num_threads, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, target.max_num_threads)

        cfg.define_knob(f"{prefix}RA", [x for x in range(len(s[output].op.reduce_axis))])

        output_local, = s.cache_write([output], "local")

        data_slices = []
        for i in range(len(output.op.axis)):
            data_slices.append(cfg[f"{prefix}D{i}"].apply(s, output, output.op.axis[i]))

        first, second, third, fourth = [x[0] for x in data_slices], \
                                       [x[1] for x in data_slices], \
                                       [x[2] for x in data_slices], \
                                       [x[3] for x in data_slices]
        
        #print("first:", first)
        #print("second:", second)
        #print("third:", third)
        #print("fourth:", fourth)

        s[output].reorder(*(first + second + third + fourth))

        s[output_local].compute_at(s[output], third[-1])
        
        do_fuse = True
        if do_fuse:
            '''
            s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
            s[output].bind(s[output].fuse(*second), te.thread_axis("vthread"))
            s[output].bind(s[output].fuse(*third), te.thread_axis("threadIdx.x"))
            '''
            s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
            s[output].bind(s[output].fuse(*second), te.thread_axis("threadIdx.x"))
        else:
            '''
            s[output].bind(first[0], te.thread_axis("blockIdx.z"))
            s[output].bind(first[1], te.thread_axis("blockIdx.y"))
            s[output].bind(first[2], te.thread_axis("blockIdx.x"))
            s[output].bind(second[0], te.thread_axis("vthread"))
            s[output].bind(second[1], te.thread_axis("vthread"))
            s[output].bind(second[2], te.thread_axis("vthread"))
            s[output].bind(third[0], te.thread_axis("threadIdx.z"))
            s[output].bind(third[1], te.thread_axis("threadIdx.y"))
            s[output].bind(third[2], te.thread_axis("threadIdx.x"))
            '''
            s[output].bind(first[0], te.thread_axis("blockIdx.z"))
            s[output].bind(first[1], te.thread_axis("blockIdx.y"))
            s[output].bind(first[2], te.thread_axis("blockIdx.x"))
            s[output].bind(second[0], te.thread_axis("threadIdx.z"))
            s[output].bind(second[1], te.thread_axis("threadIdx.y"))
            s[output].bind(second[2], te.thread_axis("threadIdx.x"))

        i = reduce_at = cfg[f"{prefix}RA"].val
        output_local_rv_o_o, output_local_rv_o_i, output_local_rv_i = cfg[f"{prefix}R{i}"].apply(s, output_local, output_local.op.reduce_axis[reduce_at])

        local_slices = []
        for i in range(len(output_local.op.axis)):
            local_slices.append(cfg[f"{prefix}D{i}"].apply(s, output_local, output_local.op.axis[i]))
        first, second, third, fourth = [x[0] for x in local_slices], [x[1] for x in local_slices], [x[2] for x in local_slices], [x[3] for x in local_slices]
        s[output_local].reorder(*(first + second + [output_local_rv_o_o,] + third + [output_local_rv_o_i,] + fourth + [output_local_rv_i]))

        #s[output_local].vectorize(fourth[-1])

        # unroll
        cfg.define_knob(f"{prefix}S", [1, 4, 32, 512])
        #cfg.define_knob(f"{prefix}S", [1, 4, 16])
        #cfg.define_knob(f"{prefix}U", [False, True])
        cfg.define_knob(f"{prefix}U", [True])
        kernel_scope = first[0]
        s[output_local].pragma(kernel_scope, "auto_unroll_max_step", cfg[f"{prefix}S"].val)
        s[output_local].pragma(kernel_scope, "unroll_explicit", cfg[f"{prefix}U"].val)

    def antares_gpu_schedule(cfg, op):
        output = C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, _, K = get_const_tuple(A.shape)

        prefix = "MM"
        is_base2 = int(os.environ.get("BASE2", "0")) > 0

        init_threads, init_vthreads = plan_threads(output.shape, s[output].op.axis)
        input_tensors = s[output].op.input_tensors

        data_sizes, reduce_sizes = [], []
        for i, ax in enumerate(s[output].op.axis):
            output_shape = get_const_tuple(output.shape)
            # Generate tile sizes.
            if i > 0:
                candidate_sizes = [2, 4, 8, 16]
                candidate_tile_sizes = []
                for ii in candidate_sizes:
                    for jj in candidate_sizes:
                        candidate_tile_sizes.append([-1, ii, jj, 1])
            else:
                candidate_tile_sizes = [[-1, 1, 1, 1]]
            #cfg.define_split(f"{prefix}D{i}", 4096 if is_base2 else output_shape[i], num_outputs=4, filter=lambda x: x.size[-1] == 1)
            cfg.define_split(f"{prefix}D{i}", 4096 if is_base2 else output_shape[i], num_outputs=4, policy="candidate", candidate=candidate_tile_sizes)
            data_sizes.append(cfg[f"{prefix}D{i}"].size)
        for i, ax in enumerate(s[output].op.reduce_axis):
            # Generate tile sizes.
            candidate_sizes = [2, 4, 8]
            candidate_tile_sizes = []
            for ii in candidate_sizes:
                for jj in candidate_sizes:
                    candidate_tile_sizes.append([-1, ii, jj])
            #cfg.define_split(f"{prefix}R{i}", 4096 if is_base2 else K, num_outputs=3)
            cfg.define_split(f"{prefix}R{i}", 4096 if is_base2 else K, num_outputs=3, policy="candidate", candidate=candidate_tile_sizes)
            reduce_sizes.append([-1, 1, cfg[f"{prefix}R{i}"].size[1] * cfg[f"{prefix}R{i}"].size[2]])

        num_threads, num_vthreads = 1, 1
        for i in range(len(s[output].op.axis)):
            num_threads *= data_sizes[i][2]
            num_vthreads *= data_sizes[i][1] * data_sizes[i][3]

        target = tvm.target.Target.current(allow_none=False)
        assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
        assert num_threads <= target.max_num_threads, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, target.max_num_threads)

        cfg.define_knob(f"{prefix}RA", [x for x in range(len(s[output].op.reduce_axis))])

        output_local, = s.cache_write([output], "local")
        
        data_slices = [list(cfg[f"{prefix}D{i}"].apply(s, output, output.op.axis[i])) for i in range(len(output.op.axis))]
        first, second, third, fourth = [x[0] for x in data_slices], [x[1] for x in data_slices], [x[2] for x in data_slices], [x[3] for x in data_slices]
        s[output].reorder(*(first + second + third + fourth))

        s[output_local].compute_at(s[output], third[-1])

        do_fuse = True
        if do_fuse:
            fused_first = s[output].fuse(*first)
            fused_second = s[output].fuse(*second)
            fused_third = s[output].fuse(*third)
            s[output].bind(fused_first, te.thread_axis("blockIdx.x"))
            s[output].bind(fused_second, te.thread_axis("vthread"))
            s[output].bind(fused_third, te.thread_axis("threadIdx.x"))
        else:
            s[output].bind(first[0], te.thread_axis("blockIdx.z"))
            s[output].bind(first[1], te.thread_axis("blockIdx.y"))
            s[output].bind(first[2], te.thread_axis("blockIdx.x"))
            s[output].bind(second[0], te.thread_axis("vthread"))
            s[output].bind(second[1], te.thread_axis("vthread"))
            s[output].bind(second[2], te.thread_axis("vthread"))
            s[output].bind(third[0], te.thread_axis("threadIdx.x"))
            s[output].bind(third[1], te.thread_axis("threadIdx.y"))
            s[output].bind(third[2], te.thread_axis("threadIdx.z"))
        
        i = reduce_at = cfg[f"{prefix}RA"].val
        output_local_rv_o_o, output_local_rv_o_i, output_local_rv_i = cfg[f"{prefix}R{i}"].apply(s, output_local, output_local.op.reduce_axis[reduce_at])

        local_slices = [list(cfg[f"{prefix}D{i}"].apply(s, output_local, output_local.op.axis[i])) for i in range(len(output_local.op.axis))]
        first, second, third, fourth = [x[0] for x in local_slices], \
                                       [x[1] for x in local_slices], \
                                       [x[2] for x in local_slices], \
                                       [x[3] for x in local_slices]
        s[output_local].reorder(*(first + second + [output_local_rv_o_o, output_local_rv_o_i] + third + [output_local_rv_i] + fourth))

        load_stage = []
        for i, load in enumerate(input_tensors):
            #cfg.define_knob(f"{prefix}AL{i}", [0, 1])
            cfg.define_knob(f"{prefix}AL{i}", [0])
            load_stage.append(s.cache_read(load, "shared", [output_local]))
            if cfg[f"{prefix}AL{i}"].val:
                s[load_stage[-1]].storage_align(load_stage[-1].op.axis[0], 2, 1)
            s[load_stage[-1]].compute_at(s[output_local], output_local_rv_o_o)

        for i, load in enumerate(load_stage):
            fused_o = s[load].fuse(*s[load].op.axis)
            val = 1 ## cfg.define_knob(f"{prefix}V{i}", [1, 2, 4] if not attrs.backend.startswith('c-hlsl_') else [1])
            fused_o, fused_i = s[load].split(fused_o, factor=val)
            s[load].vectorize(fused_i)
            fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
            s[load].bind(fused_i, te.thread_axis("threadIdx.x"))

        # unroll
        cfg.define_knob(f"{prefix}S", [1, 4, 32, 512])
        #cfg.define_knob(f"{prefix}S", [1, 4, 16])
        #cfg.define_knob(f"{prefix}S", [1, 2, 3, 4, 16])
        #cfg.define_knob(f"{prefix}U", [False, True])
        cfg.define_knob(f"{prefix}U", [True])
        kernel_scope = first[0]
        s[output_local].pragma(kernel_scope, "auto_unroll_max_step", cfg[f"{prefix}S"].val)
        s[output_local].pragma(kernel_scope, "unroll_explicit", cfg[f"{prefix}U"].val)

    def antares_cpu_small_schedule(cfg, op):
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
        
        #print("first:", first)
        #print("second:", second)
        #print("third:", third)
        #print("fourth:", fourth)

        s[output].reorder(*(first + second + third + fourth))

        s[output_local].compute_at(s[output], third[-1])
        
        do_fuse = True
        if do_fuse:
            '''
            s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
            s[output].bind(s[output].fuse(*second), te.thread_axis("vthread"))
            s[output].bind(s[output].fuse(*third), te.thread_axis("threadIdx.x"))
            '''
            s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
            s[output].bind(s[output].fuse(*second), te.thread_axis("threadIdx.x"))
        else:
            '''
            s[output].bind(first[0], te.thread_axis("blockIdx.z"))
            s[output].bind(first[1], te.thread_axis("blockIdx.y"))
            s[output].bind(first[2], te.thread_axis("blockIdx.x"))
            s[output].bind(second[0], te.thread_axis("vthread"))
            s[output].bind(second[1], te.thread_axis("vthread"))
            s[output].bind(second[2], te.thread_axis("vthread"))
            s[output].bind(third[0], te.thread_axis("threadIdx.z"))
            s[output].bind(third[1], te.thread_axis("threadIdx.y"))
            s[output].bind(third[2], te.thread_axis("threadIdx.x"))
            '''
            s[output].bind(first[0], te.thread_axis("blockIdx.z"))
            s[output].bind(first[1], te.thread_axis("blockIdx.y"))
            s[output].bind(first[2], te.thread_axis("blockIdx.x"))
            s[output].bind(second[0], te.thread_axis("threadIdx.z"))
            s[output].bind(second[1], te.thread_axis("threadIdx.y"))
            s[output].bind(second[2], te.thread_axis("threadIdx.x"))

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
        #cfg.define_knob(f"{prefix}S", [1, 4, 32, 512])
        #cfg.define_knob(f"{prefix}S", [1, 4, 16])
        #cfg.define_knob(f"{prefix}U", [False, True])
        #cfg.define_knob(f"{prefix}U", [True])
        kernel_scope = first[0]
        s[output_local].pragma(kernel_scope, "auto_unroll_max_step", 512)
        s[output_local].pragma(kernel_scope, "unroll_explicit", True)

    def antares_gpu_small_schedule(cfg, op):
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

        #target = tvm.target.Target.current(allow_none=False)
        #assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
        #assert num_threads <= target.max_num_threads, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, target.max_num_threads)

        output_local, = s.cache_write([output], "local")
        
        data_slices = []
        data_slices.append(cfg["tile_b"].apply(s, output, output.op.axis[0]))
        data_slices.append(cfg["tile_y"].apply(s, output, output.op.axis[1]))
        data_slices.append(cfg["tile_x"].apply(s, output, output.op.axis[2]))

        first, second, third, fourth = [x[0] for x in data_slices], [x[1] for x in data_slices], [x[2] for x in data_slices], [x[3] for x in data_slices]
        s[output].reorder(*(first + second + third + fourth))

        s[output_local].compute_at(s[output], third[-1])

        do_fuse = True
        if do_fuse:
            fused_first = s[output].fuse(*first)
            fused_second = s[output].fuse(*second)
            fused_third = s[output].fuse(*third)
            s[output].bind(fused_first, te.thread_axis("blockIdx.x"))
            s[output].bind(fused_second, te.thread_axis("vthread"))
            s[output].bind(fused_third, te.thread_axis("threadIdx.x"))
        else:
            s[output].bind(first[0], te.thread_axis("blockIdx.z"))
            s[output].bind(first[1], te.thread_axis("blockIdx.y"))
            s[output].bind(first[2], te.thread_axis("blockIdx.x"))
            s[output].bind(second[0], te.thread_axis("vthread"))
            s[output].bind(second[1], te.thread_axis("vthread"))
            s[output].bind(second[2], te.thread_axis("vthread"))
            s[output].bind(third[0], te.thread_axis("threadIdx.x"))
            s[output].bind(third[1], te.thread_axis("threadIdx.y"))
            s[output].bind(third[2], te.thread_axis("threadIdx.z"))
        
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
            if False:
                s[load_stage[-1]].storage_align(load_stage[-1].op.axis[0], 2, 1)
            s[load_stage[-1]].compute_at(s[output_local], output_local_rv_o_o)

        for i, load in enumerate(load_stage):
            fused_o = s[load].fuse(*s[load].op.axis)
            val = 1
            fused_o, fused_i = s[load].split(fused_o, factor=val)
            s[load].vectorize(fused_i)
            fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
            s[load].bind(fused_i, te.thread_axis("threadIdx.x"))
            #fused_o, fused_i = s[load].split(fused_o, nparts=num_threads)
            #s[load].bind(fused_o, thread_x)

        # unroll
        kernel_scope = fourth[0]
        s[output_local].pragma(kernel_scope, "auto_unroll_max_step", 512)
        s[output_local].pragma(kernel_scope, "unroll_explicit", True)

    def small_set_schedule(cfg, op):
        do_cache_read = True
        do_cache_read_local = False
        do_cache_read_unroll = True
        do_vthread = False
        auto_unroll_max_step = 256
        #print("batch_matmul.py: no local read schedule")
        C = op.output(0)
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        _, _, K = get_const_tuple(A.shape)
        _, M, N = get_const_tuple(C.shape)

        enable_tuning = get_os_env_var_bool("TVM_ENABLE_TUNING", False)
        if enable_tuning:
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
            cfg.define_knob("vectorize", [0, 1])
            #cfg.define_knob("cache_read", [0, 1])
        else:
            #tile_size = [64, 8, 64, 8, 8]
            #tile_size = [32, 8, 32, 4, 4]
            #tile_size = [96, 16, 96, 8, 6]
            tile_size = [128, 4, 16, 4, 4]
            mb_cadidates = [tile_size[0]]
            kb_cadidates = [tile_size[1]]
            nb_cadidates = [tile_size[2]]
            mr_cadidates = [tile_size[3]]
            nr_cadidates = [tile_size[4]]
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
            cfg.define_knob("vectorize", [1])
            cfg.define_knob("cache_read", [1])

        #do_cache_read = cfg["cache_read"].val
        if do_cache_read_local:
            do_cache_read = True

        if do_cache_read:
            AA = s.cache_read(A, "shared", [C])
            BB = s.cache_read(B, "shared", [C])
            if do_cache_read_local:
                AL = s.cache_read(AA, "local", [C])
                BL = s.cache_read(BB, "local", [C])
        CC = s.cache_write(C, "local")
        if op not in s.outputs:
            s[C].compute_inline()
            C = s.outputs[0].output(0)

        b, y, x = s[C].op.axis
        (k,) = s[CC].op.reduce_axis

        #ty, tx, yi, xi = s[C].tile(y, x, x_factor=cfg["mr"].val, y_factor=cfg["nr"].val)
        #by, bx, ty, tx = s[C].tile(ty, tx, x_factor=cfg["mb"].val // cfg["mr"].val, y_factor=cfg["nb"].val // cfg["nr"].val)
        by, ty, yi = cfg["tile_y"].apply(s, C, y)
        bx, tx, xi = cfg["tile_x"].apply(s, C, x)

        if cfg["vectorize"].val:
            do_vthread = False

        if do_vthread:
            auto_unroll_max_step = 0
            thread_vx = te.thread_axis("vthread", name="vx")
            thread_vy = te.thread_axis("vthread", name="vy")
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")

        s[C].reorder(b, by, bx, ty, tx, yi, xi)
        s[C].bind(b, te.thread_axis("blockIdx.z"))
        s[C].bind(by, te.thread_axis("blockIdx.y"))
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        if do_vthread:
            C = C
            s[C].bind(ty, thread_vy)
            s[C].bind(tx, thread_vx)
            s[C].bind(yi, thread_y)
            s[C].bind(xi, thread_x)
        else:
            s[C].bind(ty, thread_y)
            s[C].bind(tx, thread_x)
        if cfg["vectorize"].val:
            s[C].vectorize(xi)
        s[C].pragma(yi, "auto_unroll_max_step", auto_unroll_max_step)
        s[C].pragma(yi, "unroll_explicit", 1)

        s[CC].compute_at(s[C], tx)
        
        #ko, ki = s[CC].split(k, cfg["kb"].val)
        ko, ki = cfg["tile_k"].apply(s, CC, k)
        do_vthread = False
        if do_vthread:
            b, y, x = s[CC].op.axis
            by, ty, yi = cfg["tile_y"].apply(s, CC, y)
            bx, tx, xi = cfg["tile_x"].apply(s, CC, x)
            s[CC].reorder(b, by, bx, ty, tx, ko, ki, yi, xi)
            s[CC].bind(ty, thread_vy)
            s[CC].bind(tx, thread_vx)
            s[CC].bind(yi, thread_y)
            s[CC].bind(xi, thread_x)
        else:
            _, yi, xi = s[CC].op.axis
            s[CC].reorder(ko, ki, yi, xi)
        if cfg["vectorize"].val:
            s[CC].vectorize(xi)
        cc_unroll_axis = ki
        s[CC].pragma(cc_unroll_axis, "auto_unroll_max_step", auto_unroll_max_step)
        s[CC].pragma(cc_unroll_axis, "unroll_explicit", 1)

        if do_cache_read:
            s[AA].compute_at(s[CC], ko)
            s[BB].compute_at(s[CC], ko)
            if do_cache_read_local:
                s[AL].compute_at(s[CC], ki)
                s[BL].compute_at(s[CC], ki)
            _, y, k = s[AA].op.axis
            ty, yi = s[AA].split(y, nparts=cfg["tile_y"].size[-2])
            tx, ki = s[AA].split(k, nparts=cfg["tile_x"].size[-2])
            s[AA].reorder(ty, tx, yi, ki)
            s[AA].bind(ty, thread_y)
            s[AA].bind(tx, thread_x)
            if do_cache_read_unroll:
                s[AA].pragma(yi, "auto_unroll_max_step", auto_unroll_max_step)
                s[AA].pragma(yi, "unroll_explicit", 1)

            _, x, k = s[BB].op.axis
            ty, xi = s[BB].split(x, nparts=cfg["tile_y"].size[-2])
            tx, ki = s[BB].split(k, nparts=cfg["tile_x"].size[-2])
            s[BB].reorder(ty, tx, xi, ki)
            s[BB].bind(ty, thread_y)
            s[BB].bind(tx, thread_x)
            if do_cache_read_unroll:
                s[BB].pragma(xi, "auto_unroll_max_step", auto_unroll_max_step)
                s[BB].pragma(xi, "unroll_explicit", 1)

    def _callback(op):
        if "batch_matmul" in op.tag:
            space_name = os.getenv("TVM_TUNING_SPACE_NAME")
            if space_name is None or space_name in ["default", "large"]:
                default_schedule(cfg, op)
                #no_local_read_schedule(cfg, op)
                #no_shared_read_schedule(cfg, op)
            elif space_name == "mcpu":
                mcpu_schedule(cfg, op)
            elif space_name == "antares-cpu":
                antares_cpu_schedule(cfg, op)
            elif space_name == "antares-gpu":
                antares_gpu_schedule(cfg, op)
            elif space_name == "antares-gpu-small":
                antares_gpu_small_schedule(cfg, op)
            elif space_name == "small":
                #small_set_schedule(cfg, op)
                #antares_cpu_small_schedule(cfg, op)
                antares_gpu_small_schedule(cfg, op)

    traverse_inline(s, outs[0].op, _callback)
    return s
