
import os
from ..utils import get_os_env_var_bool, get_os_env_var_int


def verify_dense_hw_occu(target_name, task_args, config_features):
    data_bytes = 4

    if "wasm" in target_name or "llvm" in target_name:
        max_simd128_registers = get_os_env_var_int("CPU_SIMD128_REGISTERS", -1)
        max_l1_cache_size = get_os_env_var_int("CPU_MAX_L1_CACHE_SIZE", -1)
        min_l1_cache_occu = 90

        #print("verify.py: config_features {}".format(config_features))
        #input("verify.py: paused")

        if len(config_features) == 5:
            mb = int(config_features[0])
            kb = int(config_features[1])
            nb = int(config_features[2])
            mr = int(config_features[3])
            nr = int(config_features[4])
        elif len(config_features) == 8 or len(config_features) == 10:
            mb = int(config_features[1]) * int(config_features[2])
            kb = int(config_features[7])
            nb = int(config_features[4]) * int(config_features[5])
            mr = int(config_features[2])
            nr = int(config_features[5])
        
        if mr > mb or nr > nb:
            return False

        return True

        simd128_registers = mr * (nr // 4) + mr + (nr // 4)
        l1_cache_size = (mb * kb + kb * nb) * data_bytes
        l1_cache_occu = l1_cache_size * 100.0 / max_l1_cache_size
        if max_simd128_registers > 0 and simd128_registers > max_simd128_registers:
            return False
        if max_l1_cache_size > 0:
            if l1_cache_size > max_l1_cache_size:
                return False
            if l1_cache_occu < min_l1_cache_occu:
                return False
    
    return True


def verify_matmul_hw_occu(target_name, task_args, config_features):
    data_bytes = 4

    #print("verify.py: task_args %s" % str(task_args))
    #print("verify.py: m %d, n %d" % (task_args[2][1], task_args[2][2]))

    if len(config_features) <= 6:
        mb = int(config_features[0])
        kb = int(config_features[1])
        nb = int(config_features[2])
        mr = int(config_features[3])
        nr = int(config_features[4])
    elif len(config_features) <= 9:
        mb = int(config_features[1]) * int(config_features[2])
        kb = int(config_features[7])
        nb = int(config_features[4]) * int(config_features[5])
        mr = int(config_features[2])
        nr = int(config_features[5])
    elif len(config_features) <= 12:
        mb = int(config_features[1]) * int(config_features[2]) * int(config_features[3])
        kb = int(config_features[10])
        nb = int(config_features[5]) * int(config_features[6]) * int(config_features[7])
        mr = int(config_features[1]) * int(config_features[3])
        nr = int(config_features[5]) * int(config_features[7])
    elif len(config_features) <= 16:
        mb = int(config_features[5]) * int(config_features[6]) * int(config_features[7])
        kb = int(config_features[14])
        nb = int(config_features[9]) * int(config_features[10]) * int(config_features[11])
        mr = int(config_features[5]) * int(config_features[7])
        nr = int(config_features[9]) * int(config_features[11])
    if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
        print("verify.py: %s" % str([mb, kb, nb, mr, nr]))

    if "wasm" in target_name or "llvm" in target_name:
        max_simd128_registers = get_os_env_var_int("CPU_SIMD128_REGISTERS", -1)
        max_l1_cache_size = get_os_env_var_int("CPU_MAX_L1_CACHE_SIZE", -1)
        min_simd128_registers_occu = 80
        min_l1_cache_occu = 90
        
        if mr > mb or nr > nb:
            return False

        simd128_registers = mr * (nr // 4) + mr + (nr // 4)
        simd128_registers_occu = simd128_registers * 100.0 / max_simd128_registers
        
        l1_cache_size = (mb * kb + kb * nb) * data_bytes
        l1_cache_occu = l1_cache_size * 100.0 / max_l1_cache_size
        if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
            print("verify.py: simd128_registers %d" % simd128_registers)
            print("verify.py: l1_cache_size %d" % l1_cache_size)
            print("verify.py: max_l1_cache_size %d" % max_l1_cache_size)
            print("verify.py: l1_cache_occu %f" % l1_cache_occu)
        
        if max_simd128_registers > 0:
            if simd128_registers > max_simd128_registers:
                return False
            if simd128_registers_occu < min_simd128_registers_occu:
                return False
        if max_l1_cache_size > 0:
            if l1_cache_size > max_l1_cache_size:
                return False
            if l1_cache_occu < min_l1_cache_occu:
                return False
    elif target_name == "webgpu":
        max_l1_cache_size_per_processor = get_os_env_var_int("GPU_MAX_L1_CACHE_SIZE", 128 * 1024)
        max_registers_per_thread = get_os_env_var_int("GPU_REGISTERS_PER_THREAD", 255)
        max_registers_per_block = get_os_env_var_int("GPU_REGISTERS_PER_BLOCK", 65536)
        max_registers_per_processor = get_os_env_var_int("GPU_REGISTERS_PER_PROCESSOR", 65536)
        warp_size = get_os_env_var_int("GPU_WARP_SIZE", 32)
        max_active_warps = get_os_env_var_int("GPU_MAX_ACTIVE_WARPS", 48)
        max_threads_per_block = get_os_env_var_int("GPU_MAX_THREADS_PER_BLOCK", 1536)
        max_blocks_per_processor = get_os_env_var_int("GPU_MAX_BLOCKS_PER_PROCESSOR", 16)
        max_processors = get_os_env_var_int("GPU_MAX_PROCESSORS", 16)
        gpu_exec_model = os.getenv("GPU_EXECUTION_MODEL")
        gpu_shared_memory_level = get_os_env_var_int("GPU_SHARED_MEMORY_LEVEL", 3)
        if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
            print("verify.py: gpu_exec_model %s" % gpu_exec_model)
            print("verify.py: gpu_shared_memory_level %d" % gpu_shared_memory_level)
            #input("verify.py: paused")

        x_elems = task_args[2][1]
        y_elems = task_args[2][2]
        assert x_elems > 0 and y_elems > 0
        x_elem_per_workgroup = mb
        k_elem_per_workgroup = kb
        y_elem_per_workgroup = nb

        x_elem_per_thread = mr
        k_elem_per_thread = k_elem_per_workgroup
        y_elem_per_thread = nr
        use_shared_memory = 1
        if len(config_features) == 6:
            use_simd = int(config_features[5])
        elif len(config_features) == 9:
            use_simd = int(config_features[8])
        elif len(config_features) == 12:
            use_simd = int(config_features[11])
        elif len(config_features) == 16:
            use_simd = int(config_features[15])

        l1_cache_size_per_block = (x_elem_per_workgroup * k_elem_per_workgroup \
                + k_elem_per_workgroup * y_elem_per_workgroup) * data_bytes
        
        if x_elem_per_thread > x_elem_per_workgroup or y_elem_per_thread > y_elem_per_workgroup:
            return False
        if x_elem_per_thread != y_elem_per_thread:
            return False
        if (gpu_shared_memory_level > 1 and use_shared_memory == 1) or \
                (gpu_shared_memory_level == 1 and use_shared_memory == 0):
            return False
        if gpu_exec_model != "SIMD" and use_simd == 1:
            return False
        if gpu_exec_model == "SIMD" and use_simd == 0:
            return False
        
        registers_per_thread = x_elem_per_thread + y_elem_per_thread + x_elem_per_thread * y_elem_per_thread
        x_threads = x_elem_per_workgroup // x_elem_per_thread
        y_threads = y_elem_per_workgroup // y_elem_per_thread
        if x_threads < 4 or y_threads < 4:
            return False
        threads_per_block = x_threads * y_threads
        registers_per_block = registers_per_thread * threads_per_block
        active_blocks = max_blocks_per_processor
        active_blocks = min(active_blocks, max_registers_per_processor // registers_per_block)
        active_blocks = min(active_blocks, max_l1_cache_size_per_processor // l1_cache_size_per_block)
        active_blocks = min(active_blocks, max_threads_per_block // threads_per_block)
        l1_cache_size = l1_cache_size_per_block * active_blocks
        if warp_size > 0:
            num_active_warps = active_blocks * threads_per_block // warp_size
        else:
            num_active_warps = -1
        blocks = (x_elems // x_elem_per_workgroup) * (y_elems // y_elem_per_workgroup)
        waves = blocks * 1.0 / (max_processors * active_blocks)
        if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
            print("verify.py: xky_elem %s" % str([x_elem_per_thread, k_elem_per_thread, y_elem_per_thread]))
            print("verify.py: registers_per_thread %d" % registers_per_thread)
            print("verify.py: threads_per_block %d" % threads_per_block)
            print("verify.py: registers_per_block %d" % registers_per_block)
            print("verify.py: active_blocks %d" % active_blocks)
            print("verify.py: l1_cache_size %d" % l1_cache_size)
            print("verify.py: num_active_warps %d" % num_active_warps)
            print("verify.py: blocks %d" % blocks)
            print("verify.py: waves %.2f" % waves)
        if registers_per_thread > max_registers_per_thread:
            if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
                print("verify.py: not verified, too many registers per thread")
            return False
        if registers_per_block > max_registers_per_block:
            if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
                print("verify.py: not verified, too many registers per block")
            return False
        if registers_per_thread * threads_per_block * active_blocks > max_registers_per_block:
            if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
                print("verify.py: not verified, too many registers per processor")
            return False
        if l1_cache_size > max_l1_cache_size_per_processor:
            if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
                print("verify.py: not verified, too large l1 cache size")
            return False
        if active_blocks < (max_registers_per_processor // registers_per_block):
            if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
                print("verify.py: not verified, not bound by registers")
            return False
    
    return True


def verify_conv2d_gemm_hw_occu(target_name, task_args, config_features):
    #print("verify.py: task_args %s" % str(task_args))
    #print("verify.py: config_features %s" % str(config_features))
    #print("verify.py: m %d, n %d" % (task_args[2][1], task_args[2][2]))
    
    b, ih, iw, ic = task_args[0][1]
    kh, kw, _, oc = task_args[1][1]
    sh, sw = task_args[2]
    pt, pl, pb, pr = task_args[3]
    dh, dw = task_args[4]

    assert dh == 1 and dw == 1

    oh = (ih + pt + pb - kh) // sh + 1
    ow = (iw + pl + pr - kw) // sw + 1

    mb = int(config_features[1]) * int(config_features[2])
    nb = int(config_features[4]) * int(config_features[5])
    kb = int(config_features[7])

    #print("verify.py: output_shape %s" % str((b, oh, ow, oc)))

    m = oh * ow
    k = kh * kw * ic
    n = oc

    if m % mb != 0:
        m = m + (mb - m % mb)
    if k % kb != 0:
        k = k + (kb - k % kb)
    if n % nb != 0:
        n = n + (nb - n % nb)

    task_args = ((b, m, k), (b, k, n), (b, m, n))
    #print("verify.py: task_args %s" % str(task_args))
    
    return verify_matmul_hw_occu(target_name, task_args, config_features)


def verify_kernel_hw_occu(target_name, task, config_features):
    if get_os_env_var_bool("TVM_ENABLE_VERIFICATION_LOG", False):
        print("verify.py: task_name %s" % str(task.name))
        print("verify.py: config_features %s" % str(config_features))
    if "dense" in task.name:
        return verify_dense_hw_occu(target_name, task.args, config_features)
    elif "matmul" in task.name:
        return verify_matmul_hw_occu(target_name, task.args, config_features)
    elif "conv2d" in task.name:
        return verify_conv2d_gemm_hw_occu(target_name, task.args, config_features)
    else:
        raise ValueError("Unsupport kernel: " + task.name)
