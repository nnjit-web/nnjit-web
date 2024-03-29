
import os
import time
import argparse
import tvm
from tvm import relay, topi
from tvm.relay import transform, analysis
from tvm.relay.backend import Runtime
from tvm.contrib import utils, emcc
from tvm.relay.testing.init import create_workload
from kernel_tuning_common import OpType, TunerType, string_to_op_type, string_to_tuner_type
from env_tools import set_proxy_key_in_os_envs, set_target_name_in_os_envs, get_os_env_var_bool
from backend_tools import TargetBackend, is_wasm_backend, is_webgpu_backend, string_to_backend, backend_to_string
from device_info import get_processor_info
from device_capability import set_device_capability_in_os_envs
from relay_kernels.matmul import build_matmul_workload
from relay_kernels.batch_matmul import build_batch_matmul_workload, build_packed_batch_matmul_workload
from relay_kernels.dense import build_dense_workload, build_packed_dense_workload
from relay_kernels.conv2d import build_conv2d_workload
from relay_kernels.cumsum import build_cumsum_workload
from relay_kernels.softmax import build_softmax_workload


KERNEL_TUNING_CONFIGS = {
  "enable_tune": get_os_env_var_bool("TVM_ENABLE_TUNING", True),
  "enable_log_to_file": get_os_env_var_bool("TVM_ENABLE_RESULT_TO_FILE", True),
  "enable_execute_tasks": False,
  "enable_graph_executor": False,
  "graph_execute_env": "web"
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--op-type", type=str, required=True, help="Op type")
  parser.add_argument("--op-shape", type=str, required=True, help="Op shape")
  parser.add_argument("--data-type", type=str, choices=["float32", "int8"],
                      default="float32", help="Data type")
  parser.add_argument("--tuner-type", type=str, choices=["autotvm", "autoscheduler", "fast"],
                      default="autoscheduler", help="Tuner type")
  parser.add_argument("--backend", type=str,
                      choices=["cuda", "llvm-wasm", "wasm", "tvm-webgpu", "webgpu"],
                      default="wasm", help="Backend")
  parser.add_argument("--dev-info", type=str, default="unknown", help="Device infomation")
  parser.add_argument("--tune", action="store_true", help="Enable tuning")
  parser.add_argument("--number", type=int, default=50,
                      help="Number of rounds")

  return parser.parse_args()


def build_and_tune_kernel(
    op_type,
    op_shape_str,
    dtype="int8",
    data_layout="NHWC",
    tunner_type=TunerType.AutoTVM,
    dev_info="unknown",
    target_backend=TargetBackend.LLVM_WASM,
    enable_tune=True,
    number=50):
  if op_type == OpType.MatMul:
    mod, params, network = build_matmul_workload(op_shape_str, dtype)
  elif op_type == OpType.BatchMatMul:
    mod, params, network = build_batch_matmul_workload(op_shape_str, dtype)
  elif op_type == OpType.PackedBatchMatMul:
    mod, params, network = build_packed_batch_matmul_workload(op_shape_str, dtype)
  elif op_type == OpType.Dense:
    mod, params, network = build_dense_workload(op_shape_str, dtype)
  elif op_type == OpType.PackedDense:
    mod, params, network = build_packed_dense_workload(op_shape_str, dtype)
  elif op_type == OpType.Conv2d:
    data_layout = "NHWC"
    kernel_layout = "HWIO"
    #if tunner_type == TunerType.AutoTVM and target_backend == TargetBackend.WASM:
    #  data_layout = "NCHW"
    #  kernel_layout = "OIHW"
    mod, params, network = build_conv2d_workload(op_shape_str, data_layout, kernel_layout, dtype)
  elif op_type == OpType.CumSum:
    mod, params, network = build_cumsum_workload(op_shape_str, dtype)
  elif op_type == OpType.Softmax:
    mod, params, network = build_softmax_workload(op_shape_str, dtype)
  else:
    raise ValueError("Unsupported op type: " + str(op_type))
  print("op_type:", str(op_type))
  print("op_shape_str:", op_shape_str)
  print("dtype:", dtype)

  use_sparse = False
  batch_size = 1
  layout = data_layout
  runtime = Runtime("cpp", {"system-lib": True})
  if is_wasm_backend(target_backend) or is_webgpu_backend(target_backend):
    llvm_opt_level = os.getenv("LLVM_OPT_LEVEL")
    llvm_opt_level = "3" if llvm_opt_level is None else llvm_opt_level
    target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm -opt-level=" + llvm_opt_level
  elif target_backend == TargetBackend.CUDA:
    target_host = tvm.target.cuda(model="nvidia/geforce-rtx-3050", arch="sm_86")
  else:
    target_host = backend_to_string(target_backend)
  #if op_type == OpType.Conv2d and is_wasm_backend(target_backend):
  #  target_host = target_host + " -keys=arm_cpu"
  use_simd = os.getenv("EMCC_USE_SIMD")
  use_simd = (use_simd is not None) and (use_simd == "1")
  use_simd = False if target_backend == TargetBackend.CUDA else use_simd
  # -mattr=+simd128,+relaxed-simd
  target_host = target_host + " -mattr=+simd128" if use_simd else target_host
  if is_webgpu_backend(target_backend):
    target = tvm.target.Target("webgpu", host=target_host)
  elif target_backend == TargetBackend.WASM:
    target = "wasm"
  else:
    target = target_host
  target_kind_name = "llvm"
  if is_wasm_backend(target_backend) or is_webgpu_backend(target_backend):
    if not tvm.runtime.enabled(target_host):
      raise RuntimeError("Target %s is not enbaled" % target_host)

  set_target_name_in_os_envs(backend_to_string(target_backend))
  set_proxy_key_in_os_envs(dev_info)
  processor_info = get_processor_info(dev_info, target_backend)
  set_device_capability_in_os_envs(dev_info)

  timeout = 60
  repeat = 1
  number = 32
  num_measure_trials = 100000
  if tunner_type == TunerType.Fast:
    # 1000, 10000, 100000, 1000000000
    num_measure_trials = 1000
  if is_webgpu_backend(target_backend):
    if number % 32 != 0:
      #number = ((number // 32) + 1) * 32
      number = number
      #raise ValueError("Rounds must be 32 * n for WebGPU")
  if is_wasm_backend(target_backend):
    repeat = number
    number = 1
  print("dev_info:", dev_info)
  print("target:", target)
  print("processor_info:", processor_info)
  print("timeout:", timeout)
  print("repeat:", repeat)
  print("number:", number)
  print("num_measure_trials:", num_measure_trials)

  if tunner_type in [TunerType.AutoTVM, TunerType.Fast]:
    if target_backend == TargetBackend.LLVM_WASM or is_webgpu_backend(target_backend):
      build_func = "emscripten"
    elif target_backend == TargetBackend.WASM:
      build_func = "wasm"
    else:
      build_func = "default"
    n_parallel = 1
    #repeat = 1
    #number = 100
    #num_measure_trials = 1
    
    from tvm import autotvm
    builder = autotvm.LocalBuilder(
        build_func=build_func,
        runtime=runtime,
        n_parallel=n_parallel,
        do_fork=True,
        timeout=timeout
    )
    
    min_repeat_ms = 0
    if is_wasm_backend(target_backend) or is_webgpu_backend(target_backend):
      module_loader = autotvm.measure.measure_methods.WasmModuleLoader()
    else:
      module_loader = None
    
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
        module_loader=module_loader,
    )
    
    #target_backend = TargetBackend.WASM
    tuning_records_filename = "logs/%s-autotuning-%s-%s.json" % (
        network, backend_to_string(target_backend), dev_info)
    print("tuning_records_filename:", tuning_records_filename)

    tuner_name = "xgb" if tunner_type == TunerType.AutoTVM else "grid_search"
    tuning_option = {
        "tuner": tuner_name,
        "trials": num_measure_trials,
        "early_stopping": None,  # 100
        "measure_option": autotvm.measure_option(builder=builder, runner=runner),
        "tuning_records": tuning_records_filename,
    }
    print("tuning_option:", tuning_option)
    #input("\nPress ENTER key to continue")
    #time.sleep(3)

    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    enable_tune = KERNEL_TUNING_CONFIGS["enable_tune"]
    if enable_tune:
      from tvm.autotvm.tuner import XGBTuner, GridSearchTuner, RandomTuner
      for i, task in enumerate(tasks):
        print("task_space: i %d, len %d" % (i, len(tasks[0].config_space)))
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        if tuning_option["tuner"] == "xgb":
          tuner = XGBTuner(task, plan_size=64, loss_type="rank", num_threads=4)
          #tuner.update_from_file(tuning_option["tuning_records"])
        else:
          tuner = GridSearchTuner(task)
          #tuner = RandomTuner(task)
        tuner.set_error_threshold(num_measure_trials)
        callbacks = [autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix)]
        if KERNEL_TUNING_CONFIGS["enable_log_to_file"]:
          callbacks.append(autotvm.callback.log_to_file(tuning_option["tuning_records"]))

        tuner.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=callbacks
        )

    enable_execute_tasks = KERNEL_TUNING_CONFIGS["enable_execute_tasks"]
    if enable_execute_tasks:
      #dev_info = "honor-magicbook-16"
      set_proxy_key_in_os_envs(dev_info)
      print("exec_dev_info:", dev_info)
      # Execute with best configs.
      from tvm.autotvm.runner import Runner
      for i, task in enumerate(tasks):
        key = (task.target.keys[0], task.workload)
        config = autotvm.pick_best_config_by_targetkey(tuning_option["tuning_records"], key)
        runner = Runner(task, config, tuning_option["measure_option"])
        runner.run()

    enable_graph_executor = KERNEL_TUNING_CONFIGS["enable_graph_executor"]
    if enable_graph_executor:
      if not os.path.exists(tuning_option["tuning_records"]):
        with open(tuning_option["tuning_records"], "w") as f:
          f.write("")

      exec_env = KERNEL_TUNING_CONFIGS["graph_execute_env"]
      # Compile the optimized model with tuning data.
      with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
          lib = relay.build(mod, target=target, runtime=runtime, params=params)

      if exec_env == "local":
        dev = tvm.device(str(target), 0)
        #print("lib:", str(lib))
        #print("lib[\"default\"]:", str(lib["default"]))
        #print("dev:", str(dev))
        #print("lib[\"default\"](dev):", str(lib["default"](dev)))
        m = graph_executor.GraphModule(lib["default"](dev))
        m.run()
      elif exec_env == "web":
        wasm_path = "dist/wasm/compiled_relay_lib.wasm"
        lib.get_lib().export_library(wasm_path, emcc.create_tvmjs_wasm)
        wasm_binary = open(wasm_path, "rb").read()

        from tvm import rpc
        proxy_host = "127.0.0.1"
        proxy_port = 9090
        key = dev_info
        session_timeout = 10
        remote = rpc.connect(
            proxy_host,
            proxy_port,
            key=key,
            session_timeout=session_timeout,
            session_constructor_args=["rpc.WasmSession", wasm_binary],
        )

        # Execute model.
        import numpy as np
        from tvm.contrib import graph_executor
        dev = remote.device(str(target), 0)
        graph_json = lib.get_graph_json()
        print("graph_json:", graph_json)

        enable_debug = False
        if not enable_debug:
          fcreate = remote.get_function("__sync.wasm.createGraphExecutor")
          print("create graph executor ...")
          fcreate(graph_json, dev)
          
          #time.sleep(3)
          
          rounds = 10
          print("execute graph executor ...")
          if is_wasm_backend(target_backend):
            frun = remote.get_function("__sync.wasm.runGraphExecutorForWasm")
            cost_bytes = frun(dev, rounds)
          elif is_webgpu_backend(target_backend):
            frun = remote.get_function("__sync.wasm.runGraphExecutorForWebGPU")
            fisfinished = remote.get_function("__sync.wasm.isTimeExecutionForWebGPUFinished")
            fgetcost = remote.get_function("__sync.wasm.getTimeExecutionForWebGPUResults")
            frun(dev, rounds)
            while fisfinished() == 0:
              time.sleep(1)
            cost_bytes = fgetcost()
          costs = np.frombuffer(cost_bytes, dtype=np.dtype("<f8"))
          print("costs:", costs)
        else:
          fcreate = remote.get_function("__sync.wasm.createGraphExecutorDebug")
          frunindividual = remote.get_function("__sync.wasm.runIndividualGraphExecutorDebug")

          print("create graph executor debug ...")
          fcreate(graph_json, dev)
          print("profile graph executor debug ...")
          num_rounds = 10
          repeat = 1
          min_repeat_ms = 0
          limit_zero_time_iterations = 0
          cooldown_interval_ms = 0
          repeats_to_cooldown = 1
          report_json = frunindividual(num_rounds,
                                       repeat,
                                       min_repeat_ms,
                                       limit_zero_time_iterations,
                                       cooldown_interval_ms,
                                       repeats_to_cooldown)
          print("report_json:", report_json)
  elif tunner_type == TunerType.AutoScheduler:
    log_file = "logs/%s-%s-B%d-%s-%s.json" % (
        network, layout, batch_size, processor_info, backend_to_string(target_backend))

    from tvm import auto_scheduler
    hardware_params = None
    if is_webgpu_backend(target_backend):
      max_shared_memory_per_block = 64 * 1024 * 1024;
      max_threads_per_block = 24
      '''
      hardware_params = auto_scheduler.HardwareParams(
          num_cores=-1,
          vector_unit_bytes=16,
          cache_line_bytes=64,
          max_shared_memory_per_block=int(max_shared_memory_per_block),
          max_threads_per_block=int(max_threads_per_block),
          # The value `max_local_memory_per_block` is not used in AutoScheduler,
          # but is required by the API.
          max_local_memory_per_block=12345678,
          max_vthread_extent=8,
          warp_size=32
      )
      '''
      hardware_params = auto_scheduler.HardwareParams(
          max_shared_memory_per_block=int(max_shared_memory_per_block),
          target=target,
      )
    
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, hardware_params=hardware_params)
    for idx, task in enumerate(tasks):
      print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
      print(task.compute_dag)

    if enable_tune:
      print("Begin tuning...")
      n_parallel = 1
      #import multiprocessing
      #n_parallel = multiprocessing.cpu_count()
      builder = auto_scheduler.LocalBuilder(timeout=timeout,
                                            n_parallel=n_parallel,
                                            build_func="emscripten")
      runner = auto_scheduler.LocalRunner(number=number,
                                          repeat=repeat,
                                          timeout=timeout,
                                          enable_cpu_cache_flush=True)
      tune_option = auto_scheduler.TuningOptions(
          num_measure_trials=num_measure_trials,  # change this to 20000 to achieve the best performance
          builder=builder,
          runner=runner,
          measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
      )

      tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

      if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
      else:
        tuner.tune(tune_option)
    else:
      print("Resume tuning...")
      cost_model = auto_scheduler.XGBModel()
      cost_model.update_from_file(log_file)
      search_policy = [
          auto_scheduler.SketchPolicy(
              task,
              cost_model,
              init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
          ) for task in tasks
      ]
      tune_option = auto_scheduler.TuningOptions(
          num_measure_trials=num_measure_trials,
          measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
      )
      task.tune(tune_option, search_policy=search_policy)
    
      '''
      # Compile with the history best
      print("Compile...")
      with auto_scheduler.ApplyHistoryBest(log_file):
          with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
              lib = relay.build(mod, target=target, params=params)
      '''
      for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
        sch, args = task.apply_best(log_file)
        print("Lowered TIR:")
        print(tvm.lower(sch, args, simple_mode=True))
        print("Equivalent Schedule:")
        print(task.print_best(log_file))
      
      #run_resume_tuning()
  else:
    raise ValueError("Not supported tuner type: " + tunner_type)


if __name__ == "__main__":
  args = parse_args()
  build_and_tune_kernel(
    op_type=string_to_op_type(args.op_type),
    op_shape_str=args.op_shape,
    dtype=args.data_type,
    tunner_type=string_to_tuner_type(args.tuner_type),
    dev_info=args.dev_info,
    target_backend=string_to_backend(args.backend),
    enable_tune=args.tune,
    number=args.number
  )
