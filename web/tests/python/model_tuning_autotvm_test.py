
import os
import time
import argparse
import pickle
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor, emcc
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GridSearchTuner
from tvm import autotvm
from tvm.relay.backend import Runtime
import tvm.relay.testing
from relay_models.roberta import roberta_model
from relay_models.bart import bart_model
from relay_models.gpt2 import gpt2_model
from relay_models.t5 import t5_encoder_model
from env_tools import set_proxy_key_in_os_envs
from backend_tools import is_wasm_backend, is_webgpu_backend
from device_capability import set_device_capability_in_os_envs


MODEL_TUNING_CONFIGS = {
    "enable_save_relay_model": True,
    "enable_skip_load_relay_model": False,
    "enable_tune": True,
    "enable_log_to_file": True,
    "repeat": 50,
    "enable_execute_tasks": False,
    "enable_graph_executor": False,
    "enable_graph_executor_run": False,
    "graph_execute_env": "web",
    "enable_graph_executor_timeline_test": False
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--dev-info", type=str, required=True, help="Device info")
    parser.add_argument("--backend", type=str, required=True, help="Backend")
    parser.add_argument("--num-trails", type=int, default=1000, help="Number of trails")
    return parser.parse_args()


def get_mod_and_params(model_name, dtype="float32"):
    if model_name == "roberta":
        return roberta_model(dtype)
    elif model_name == "bart":
        return bart_model()
    elif model_name == "gpt-2":
        return gpt2_model()
    elif model_name == "t5-encoder":
        return t5_encoder_model()
    else:
        raise ValueError("Unsupported model:", model_name)


def get_use_simd():
    use_simd = os.getenv("EMCC_USE_SIMD")
    return (use_simd is not None) and (use_simd == "1")


def get_target_and_bulild_func(backend):
    if backend == "llvm-wasm":
        target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm"
        if get_use_simd():
            target_host += " -mattr=+simd128"
            target = target_host
            build_func = "emscripten"
    elif is_webgpu_backend(backend):
        target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm"
        if get_use_simd():
            target_host += " -mattr=+simd128"
            target = tvm.target.Target("webgpu", host=target_host)
            build_func = "emscripten"
    elif backend == "wasm":
        target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm"
        if get_use_simd():
            target_host += " -mattr=+simd128"
            target = "wasm"
            build_func = "wasm"
    else:
        raise ValueError("Unsupported backend:", backend)
    return target_host, target, build_func


def get_number_and_repeat(backend):
    if is_wasm_backend(backend):
        return 1, 50
    elif is_webgpu_backend(backend):
        return 64, 1
    else:
        raise ValueError("Unsupported backend:", backend)


def model_tuning_test(model_name, dev_info, backend="llvm-wasm", num_trails=1000):
    runtime = Runtime("cpp", {"system-lib": True})
    target_host, target, build_func = get_target_and_bulild_func(backend)
    if not tvm.runtime.enabled(target_host):
        raise RuntimeError("Target %s is not enbaled" % target)

    dtype = "float32"

    #processor_info = get_processor_info(dev_info, backend)
    set_proxy_key_in_os_envs(dev_info)
    set_device_capability_in_os_envs(dev_info)

    number, repeat = get_number_and_repeat(backend)
    if MODEL_TUNING_CONFIGS["repeat"] is not None:
        if is_wasm_backend(backend):
            repeat = MODEL_TUNING_CONFIGS["repeat"]
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
    timeout = 60  # in seconds

    print("model_name:", model_name)
    print("dtype:", dtype)
    print("dev_info:", dev_info)
    print("backend:", backend)
    print("number:", number)
    print("repeat:", repeat)

    enable_tune = MODEL_TUNING_CONFIGS["enable_tune"]
    enable_execute_tasks = MODEL_TUNING_CONFIGS["enable_execute_tasks"]
    enable_graph_executor = MODEL_TUNING_CONFIGS["enable_graph_executor"]
    enable_graph_executor_timeline_test = MODEL_TUNING_CONFIGS["enable_graph_executor_timeline_test"]

    print("enable_tune:", enable_tune)
    print("enable_graph_executor:", enable_graph_executor)
    print("enable_execute_tasks:", enable_execute_tasks)

    time.sleep(2)

    if enable_graph_executor or enable_graph_executor_timeline_test:
        if is_wasm_backend(backend):
            target_host, target, build_func = get_target_and_bulild_func("llvm-wasm")

    # Create TVM builder and runner.
    builder = autotvm.LocalBuilder(
        build_func=build_func,
        runtime=runtime,
        n_parallel=1,
        do_fork=True,
        timeout=timeout
    )

    module_loader = autotvm.measure.measure_methods.WasmModuleLoader()
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
        module_loader=module_loader,
    )

    opt_level = os.getenv("EMCC_OPT_LEVEL")
    opt_level = "-O3" if opt_level is None else opt_level
    use_simd = "simd" if get_use_simd() else "nosimd"
    compiler_info = opt_level[1:].lower() + "-" + use_simd

    logs_dirname = "logs"
    if not os.path.exists(logs_dirname):
        os.makedirs(logs_dirname)

    tuning_records_filename = os.path.join(logs_dirname, "%s-autotuning-%s-%s-%s.json" % (model_name, backend, dev_info, compiler_info))
    #tuning_records_filename = "logs/%s-autotuning-temp.json" % (model_name)
    print("tuning_records_filename:", tuning_records_filename)

    engine_name = os.getenv("ENGINE")
    if engine_name == "nnjit":
        num_trails = 100000
        tuner_name = "grid_search"
    elif engine_name == "tvm":
        num_trails = 1000
        tuner_name = "xgb"
    tuning_option = {
        "tuner": tuner_name,
        "trials": num_trails,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(builder=builder, runner=runner),
        "tuning_records": tuning_records_filename,
    }
    print("tuning_option:", tuning_option)

    time.sleep(3)

    enable_skip_load_relay_model = MODEL_TUNING_CONFIGS["enable_skip_load_relay_model"]
    if not enable_skip_load_relay_model:
        enable_save_relay_model = MODEL_TUNING_CONFIGS["enable_save_relay_model"]
        mod_filepath = "relay_models/" + model_name + "-" + backend + ".mod"
        params_filepath = "relay_models/" + model_name + "-" + backend + ".params"
        start_time = time.time()
        if enable_save_relay_model and \
            os.path.exists(mod_filepath) and os.path.exists(params_filepath):
            print("load relay model ...")
            with open(mod_filepath, "r") as f:
                print("load " + mod_filepath)
                mod = tvm.ir.load_json(f.read())
            with open(params_filepath, "rb") as f:
                print("load " + params_filepath)
                params = relay.load_param_dict(f.read())
            print("load relay model: %.3f sec" % (time.time() - start_time))
        else:
            print("parse model ...")
            mod, params = get_mod_and_params(model_name, dtype)
            print("parse model: %.3f sec" % (time.time() - start_time))

            if enable_save_relay_model:
                if not os.path.exists("relay_models"):
                    os.makedirs("relay_models")
                with open(mod_filepath, "w") as f:
                    print("save " + mod_filepath)
                    f.write(tvm.ir.save_json(mod))
                with open(params_filepath, "wb") as f:
                    print("save " + params_filepath)
                    f.write(relay.save_param_dict(params))

    if not enable_skip_load_relay_model:
        # Begin by extracting the tasks from the onnx model.
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
        for i, task in enumerate(tasks):
            print("task: idx %d, workload %s" % (i, task.workload))

        #target_task_idx = None
        #target_task_idx = [4, 5]
        #target_task_idx = [len(tasks) - 1]
        target_task_idx = list(range(0, len(tasks)))
        print("target_task_idx:", target_task_idx)
        #input("Press ENTER to continue")
        time.sleep(3)

    if enable_skip_load_relay_model:
        enable_tune = False
        enable_execute_tasks = False

    if enable_tune:
        os.environ["TVM_EXEC_TYPE"] = "kernel"
        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            if (target_task_idx is not None) and (i not in target_task_idx):
                continue

            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            if tuning_option["tuner"] == "xgb":
                tuner_obj = XGBTuner(task, plan_size=64, loss_type="rank", num_threads=4)
            else:
                tuner_obj = GridSearchTuner(task)
                tuner_obj.set_error_threshold(num_trails)
            callbacks = [autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix)]
            if MODEL_TUNING_CONFIGS["enable_log_to_file"]:
                callbacks.append(autotvm.callback.log_to_file(tuning_option["tuning_records"]))
            tuner_obj.tune(
                n_trial=min(tuning_option["trials"], len(task.config_space)),
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=callbacks,
            )

    if enable_execute_tasks:
        os.environ["TVM_EXEC_TYPE"] = "kernel"
        execute_dev_info = dev_info
        #execute_dev_info = "honor-70"
        set_proxy_key_in_os_envs(execute_dev_info)
        print("execute_dev_info:", execute_dev_info)

        # Execute with best configs.
        from tvm.autotvm.runner import Runner
        total_result_text = ""
        total_mean_cost = 0
        for i, task in enumerate(tasks):
            if (target_task_idx is not None) and (i not in target_task_idx):
                continue
            for target_key in task.target.keys:
                if target_key != "cpu" and target_key != "webgpu":
                    continue
                key = (target_key, task.workload)
                config = autotvm.pick_best_config_by_targetkey(tuning_option["tuning_records"], key)
                print("task %d, best_config %s" % (i, str(config)))

                runner = Runner(task, config, tuning_option["measure_option"])
                costs = runner.run()
                try:
                    mean_cost = np.mean(costs)
                except TypeError:
                    mean_cost = 1e9

                result_text = "task %d, target_key %s, mean_cost %.6f" % (i, target_key, mean_cost)
                print(result_text)
                total_result_text += result_text + "\n"
                total_mean_cost += mean_cost
        print("total_result_text:")
        print(total_result_text)
        print("target_key %s, model %s, total_mean_cost %.6f" % (target_key, model_name, total_mean_cost))

    if enable_graph_executor:
        os.environ["TVM_EXEC_TYPE"] = "model"
        lib = None
        if not enable_skip_load_relay_model:
            print("relay build model ...")
        # Compile the optimized model with tuning data.
        with autotvm.apply_history_best(tuning_option["tuning_records"]):
            with tvm.transform.PassContext(opt_level=3, config={}):
                lib = relay.build(mod, target=target, runtime=runtime, params=params)

        exec_env = MODEL_TUNING_CONFIGS["graph_execute_env"]
        if exec_env == "local":
            dev = tvm.device(str(target), 0)
            module = graph_executor.GraphModule(lib["default"](dev))
            module.run()
        elif exec_env == "web":
            wasm_path = "relay_libs/%s.wasm" % model_name
            if lib is not None:
                print("export_library:", wasm_path)
                if not os.path.exists("relay_libs"):
                    os.makedirs("relay_libs")
                    lib.get_lib().export_library(wasm_path, emcc.create_tvmjs_wasm)

            if MODEL_TUNING_CONFIGS["enable_graph_executor_run"]:
                import xmlrpc.client
                # 192.168.43.98
                browser_rpc_ip = "192.168.43.98"
                browser_rpc_port = 8288
                browser_server = None
                #browser_server = xmlrpc.client.ServerProxy("http://%s:%d" % (browser_rpc_ip, browser_rpc_port))
                if browser_server is not None:
                    browser_pid = browser_server.open_chrome()

                wasm_binary = open(wasm_path, "rb").read()

                from tvm import rpc
                proxy_host = "127.0.0.1"
                proxy_port = os.getenv("PROXY_PORT")
                proxy_port = 9090 if proxy_port is None else int(proxy_port)
                key = os.getenv("PROXY_KEY")
                key = "wasm" if key is None else key
                session_timeout = 10
                print("rpc.connect, host %s, port %d, key %s" % (proxy_host, proxy_port, key))
                remote = rpc.connect(
                    proxy_host,
                    proxy_port,
                    key=key,
                    session_timeout=session_timeout,
                    session_constructor_args=["rpc.WasmSession", wasm_binary],
                )
                print("remote.device")
                dev = remote.device(str(target), 0)

                filepath = os.path.join("logs", "%s.json" % model_name)
                if lib is not None:
                    print("graph_json")
                    graph_json = lib.get_graph_json()
                    #print("graph_json:", str(graph_json))
                    enable_write_graph_json = True
                    if enable_write_graph_json:
                        with open(filepath, "w") as out:
                            out.write(graph_json)
                        print("graph_json_filepath:", filepath)
                else:
                    with open(filepath, "r") as f:
                        graph_json = f.read()

                enable_profile = False
                if not enable_profile:
                    fcreate = remote.get_function("__sync.wasm.createGraphExecutor")
                    fsetinput = remote.get_function("__sync.wasm.setGraphExecutorInput")

                    print("create graph executor ...")
                    fcreate(graph_json, dev)
                    '''
                    if model_name == "roberta" or model_name == "bart":
                        print("set graph executor input ...")
                        input_shape = (1, 384)
                        input_ids_data = np.zeros(input_shape).astype("int64")
                        attention_mask_data = np.zeros(input_shape).astype("int64")
                        #fsetinput("input_ids", tvm.nd.array(input_ids_data, dev))
                        #fsetinput("attention_mask", tvm.nd.array(attention_mask_data, dev))
                        fsetinput(0, tvm.nd.array(input_ids_data, dev))
                        fsetinput(1, tvm.nd.array(attention_mask_data, dev))
                    elif model_name == "gpt-2":
                        print("set graph executor input ...")
                        input_shape = (1, 10, 64)
                        input_ids_data = np.zeros(input_shape).astype("int64")
                        fsetinput(0, tvm.nd.array(input_ids_data, dev))
                    '''

                    print("wait 5 sec ...")
                    time.sleep(5)
                    #input("Press ENTER to continue")

                    print("run graph executor ...")
                    num_rounds = 10
                    print("num_rounds:", num_rounds)
                    if is_wasm_backend(backend):
                        frun = remote.get_function("__sync.wasm.runGraphExecutorForWasm")
                        cost_bytes = frun(dev, num_rounds)
                    elif is_webgpu_backend(backend):
                        frun = remote.get_function("__sync.wasm.runGraphExecutorForWebGPU")
                        fisfinished = remote.get_function("__sync.wasm.isTimeExecutionForWebGPUFinished")
                        fgetcost = remote.get_function("__sync.wasm.getTimeExecutionForWebGPUResults")
                        frun(dev, num_rounds)
                        while fisfinished() == 0:
                            time.sleep(1)
                        cost_bytes = fgetcost()
                    costs = np.frombuffer(cost_bytes, dtype=np.dtype("<f8"))
                    if is_webgpu_backend(backend):
                        print("costs:", costs)
                        new_costs = []
                        tstart = costs[0]
                        for i in range(1, len(costs)):
                            costs[i] = costs[i] - tstart
                            tstart = tstart + costs[i]
                            new_costs.append(costs[i])
                        costs = np.array(new_costs)

                        new_costs = []
                        ref_cost = costs[-1]
                        for i in range(0, len(costs)):
                            relative_cost = costs[i] / ref_cost
                            if relative_cost >= 0.8 and relative_cost <= 1.2:
                                new_costs.append(costs[i])
                        costs = np.array(new_costs)
                    else:
                        costs = costs[1:-1]
                    fprofilememory = remote.get_function("__sync.wasm.profileMemory")
                    memory = fprofilememory()

                    print("costs:", costs)
                    print("mean_cost:", np.mean(costs))
                    print("memory:" + str(memory) + " MiB")
                else:
                    fcreate = remote.get_function("__sync.wasm.createGraphExecutorDebug")
                    fsetinput = remote.get_function("__sync.wasm.setGraphExecutorInput")
                    frunindividual = remote.get_function("__sync.wasm.runIndividualGraphExecutorDebug")
                    #fprofile = remote.get_function("__sync.wasm.profileGraphExecutorDebug")

                    print("create graph executor debug ...")
                    fcreate(graph_json, dev)
                    if model_name == "bart":
                        print("set graph executor input ...")
                        input_ids_data = np.zeros((1, 384)).astype("int64")
                        attention_mask_data = np.zeros((1, 384)).astype("int64")
                        fsetinput(0, tvm.nd.array(input_ids_data, dev))
                        fsetinput(1, tvm.nd.array(attention_mask_data, dev))
                    print("profile graph executor debug ...")
                    num_rounds = 1
                    repeat = 2
                    min_repeat_ms = 0
                    limit_zero_time_iterations = 0
                    cooldown_interval_ms = 0
                    repeats_to_cooldown = 1
                    report_json = frunindividual(
                        num_rounds,
                        repeat,
                        min_repeat_ms,
                        limit_zero_time_iterations,
                        cooldown_interval_ms,
                        repeats_to_cooldown
                    )
                    print("report_json:", report_json)

                if browser_server is not None:
                    browser_server.close_chrome(browser_pid)

    if enable_graph_executor_timeline_test:
        print("===== Graph Executor Timeline Test =====")
        os.environ["TVM_EXEC_TYPE"] = "model"
        def build_model_by_history_index(model_name, log_file, idx, mod, target, runtime, params):
            with autotvm.apply_history_best_by_index(log_file, idx):
                with tvm.transform.PassContext(opt_level=3, config={}):
                    lib = relay.build(mod, target=target, runtime=runtime, params=params)
            wasm_path = "relay_libs/%s.wasm" % model_name
            print("export_library:", wasm_path)
            if not os.path.exists("relay_libs"):
                os.makedirs("relay_libs")
            lib.get_lib().export_library(wasm_path, emcc.create_tvmjs_wasm)
            return lib, wasm_path

        def run_model(lib, wasm_path):
            import xmlrpc.client
            browser_rpc_ip = "127.0.0.1"
            browser_rpc_port = 8288
            #browser_server = xmlrpc.client.ServerProxy("http://%s:%d" % (browser_rpc_ip, browser_rpc_port))
            browser_server = None
            if browser_server is not None:
                browser_pid = browser_server.open_chrome()
                time.sleep(5)

            wasm_binary = open(wasm_path, "rb").read()

            from tvm import rpc
            proxy_host = "127.0.0.1"
            proxy_port = os.getenv("PROXY_PORT")
            proxy_port = 9090 if proxy_port is None else int(proxy_port)
            key = os.getenv("PROXY_KEY")
            key = "wasm" if key is None else key
            session_timeout = 10
            print("rpc.connect, host %s, port %d, key %s" % (proxy_host, proxy_port, key))
            remote = rpc.connect(
                proxy_host,
                proxy_port,
                key=key,
                session_timeout=session_timeout,
                session_constructor_args=["rpc.WasmSession", wasm_binary],
            )
            print("remote.device")
            dev = remote.device(str(target), 0)

            filepath = os.path.join("logs", "%s.json" % model_name)
            if lib is not None:
                print("graph_json")
                graph_json = lib.get_graph_json()
                #print("graph_json:", str(graph_json))
                enable_write_graph_json = True
                if enable_write_graph_json:
                    with open(filepath, "w") as out:
                        out.write(graph_json)
                    print("graph_json_filepath:", filepath)
            else:
                with open(filepath, "r") as f:
                    graph_json = f.read()

            fcreate = remote.get_function("__sync.wasm.createGraphExecutor")
            fsetinput = remote.get_function("__sync.wasm.setGraphExecutorInput")

            print("create graph executor ...")
            fcreate(graph_json, dev)
            print("wait 5 sec ...")
            #time.sleep(5)
            #input("Press ENTER to continue")

            print("run graph executor ...")
            num_rounds = 5
            print("num_rounds:", num_rounds)
            if is_wasm_backend(backend):
                frun = remote.get_function("__sync.wasm.runGraphExecutorForWasm")
                cost_bytes = frun(dev, num_rounds)
            elif is_webgpu_backend(backend):
                frun = remote.get_function("__sync.wasm.runGraphExecutorForWebGPU")
                fisfinished = remote.get_function("__sync.wasm.isTimeExecutionForWebGPUFinished")
                fgetcost = remote.get_function("__sync.wasm.getTimeExecutionForWebGPUResults")
                frun(dev, num_rounds)
                while fisfinished() == 0:
                    time.sleep(1)
                cost_bytes = fgetcost()
            costs = np.frombuffer(cost_bytes, dtype=np.dtype("<f8"))
            if is_webgpu_backend(backend):
                print("costs:", costs)
                new_costs = []
                tstart = costs[0]
                for i in range(1, len(costs)):
                    costs[i] = costs[i] - tstart
                    tstart = tstart + costs[i]
                    new_costs.append(costs[i])
                costs = np.array(new_costs)
                new_costs = []
                ref_cost = costs[-1]
                for i in range(0, len(costs)):
                    relative_cost = costs[i] / ref_cost
                    if relative_cost >= 0.8 and relative_cost <= 1.2:
                        new_costs.append(costs[i])
                costs = np.array(new_costs)
            else:
                costs = costs[1:-1]
            print("costs:", costs)
            print("mean_cost:", np.mean(costs))
            if browser_server is not None:
                browser_server.close_chrome(browser_pid)
            return np.mean(costs)

        def compare_kernel_costs(costs0, costs1):
            assert len(costs0) == len(costs1)
            ret = 0
            for c0, c1 in zip(costs0, costs1):
                if c0 < c1:
                    ret = -1
                elif c0 > c1 and ret >= 0:
                    ret = 1
                elif c0 == c1 and ret == 0:
                    ret = 0
            return ret

        timeline = []
        model_costs = []
        kernel_costs = []

        cur_time = 0
        do_model_run = True
        log_file = tuning_option["tuning_records"]
        init_i = 0
        min_kernel_costs, max_cost_count = autotvm.pick_best_costs_by_targetkey(log_file, init_i)
        lib, wasm_path = build_model_by_history_index(model_name, log_file, init_i, mod, target, runtime, params)
        print("i %d, icount %d, cur_time %.3f, model_cost %.3f" % (init_i, max_cost_count, cur_time, 0))
        print("kernel_costs:", min_kernel_costs)

        for i in range(init_i + 1, max_cost_count):
            if do_model_run:
                model_cost = run_model(lib, wasm_path)
            print("i %d, icount %d, cur_time %.3f, model_cost %.3f" % (i, max_cost_count, cur_time, model_cost))
            timeline.append(cur_time)
            model_costs.append(model_cost)
            cur_time += model_cost
            kernel_costs, _ = autotvm.pick_best_costs_by_targetkey(log_file, i)
            print("kernel_costs:", kernel_costs)
            cur_time += np.sum(np.array(kernel_costs))
            if compare_kernel_costs(kernel_costs, min_kernel_costs) < 0:
                lib, wasm_path = build_model_by_history_index(model_name, log_file, i, mod, target, runtime, params)
                min_kernel_costs = kernel_costs
                do_model_run = True
            else:
                do_model_run = False

            timeline = list(np.array(timeline) * 1000.0)
            model_costs = list(np.array(model_costs) * 1000.0)
            print("timeline:", str(timeline))
            print("model_costs:", str(model_costs))


args = parse_args()
model_name = args.model_name
dev_info = args.dev_info
backend = args.backend
num_trails = args.num_trails
model_tuning_test(model_name, dev_info, backend, num_trails)
