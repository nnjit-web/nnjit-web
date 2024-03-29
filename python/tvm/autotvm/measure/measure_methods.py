# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,too-many-function-args,too-many-nested-blocks
"""
Functions that run on executor for measurement.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.
"""

import contextlib
import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
import typing
import warnings
import xmlrpc.client
from collections import namedtuple
from random import getrandbits

import tvm._ffi
import tvm.ir.transform
from tvm import nd
from tvm import rpc as _rpc
from tvm.autotvm.env import AutotvmGlobalScope, reset_global_scope
from tvm.contrib import ndk, stackvm, tar, emcc, wasm
from tvm.contrib.popen_pool import PopenPoolExecutor
from tvm.driver import build
from tvm.error import TVMError
from tvm.target import Target

from ..env import AutotvmGlobalScope
from ..task.space import InstantiationError
from ..utils import get_const_tuple, get_os_env_var_bool
from .measure import Builder, MeasureErrorNo, MeasureResult, Runner

logger = logging.getLogger("autotvm")
dev_info = os.getenv("DEV_INFO")
if dev_info is not None and False:
    if "honor-magicbook-16" in dev_info:
        browser_rpc_ip = "127.0.0.1"
    elif "surface-book-3" in dev_info:
        #browser_rpc_ip = "192.168.43.98"
        browser_rpc_ip = None
    if browser_rpc_ip is not None:
        browser_rpc_port = 8288
        browser_server = xmlrpc.client.ServerProxy("http://%s:%d" % (browser_rpc_ip, browser_rpc_port))
    else:
        browser_server = None
else:
    browser_server = None

total_kernel_count = None
built_kernel_count = None

MEASURE_METHODS_CONFIGS = {
    "enable_build": True,
    "enable_export": True,
    "enable_run": True
}

class BuildResult(namedtuple("BuildResult", ("filename", "arg_info", "error", "time_cost"))):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    filename : str
        The filename of generated library
    arg_info : Tuple
        The shape and dtype information of tvm tensor arguments
    error : Exception
        The error happens during compilation.
    time_cost : float
        The time cost of building
    """


class LocalBuilder(Builder):
    """Run compilation on local machine

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    build_kwargs: dict
        If supplied, additional kwargs passed to build_func. Overrides any build_kwargs supplied
        by the Runner.
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If id 'stackvm', use function for stackvm
        If is callable, use it as custom build function, expect lib_format field.
    do_fork: bool
        If False, do not fork when building. Requires n_parallel=1.
    runtime: Optional[Runtime]
        Specify the runtime to generate artifacts for
    """

    def __init__(
        self,
        timeout=10,
        n_parallel=None,
        build_kwargs=None,
        build_func="default",
        do_fork=False,
        runtime=None,
    ):
        super(LocalBuilder, self).__init__(timeout, n_parallel, build_kwargs)

        if isinstance(build_func, str):
            if build_func == "default":
                build_func = tar.tar
            elif build_func == "ndk":
                build_func = ndk.create_shared
            elif build_func == "stackvm":
                build_func = stackvm.build
            elif build_func == "emscripten":
                build_func = emcc.create_tvmjs_wasm
            elif build_func == "wasm":
                build_func = wasm.create_wasm
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _WrappedBuildFunc(build_func, runtime)
        if not do_fork:
            assert n_parallel in (
                None,
                1,
            ), f"if do_fork=False, need n_parallel=None or 1; got {n_parallel}"
            self.executor = None
        else:
            self.executor = PopenPoolExecutor(
                max_workers=1, timeout=timeout, initializer=reset_global_scope, initargs=(AutotvmGlobalScope.current,)
            )
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i : i + self.n_parallel]:
                if self.executor is not None:
                    ret = self.executor.submit(self.build_func, inp, self.tmp_dir, **self.build_kwargs)
                else:
                    ret = self.build_func(inp, self.tmp_dir, **self.build_kwargs)
                futures.append(ret)

            for future in futures:
                try:
                    if self.executor is not None:
                        res = future.result()
                    else:
                        res = future
                    #print("measure_methods.py: inp %s, time_cost %f" % (str(inp), res.time_cost))
                    if res.error is not None:
                        assert len(res.error) == 2, (
                            f"BuildResult errors should be a 2-tuple, but it is a {len(res.error)}"
                            "-tuple. This should not happen!"
                        )
                        tb, exception = res.error
                        # instantiation error
                        if isinstance(exception, InstantiationError):
                            res = MeasureResult(
                                (
                                    tb,
                                    exception,
                                ),
                                MeasureErrorNo.INSTANTIATION_ERROR,
                                res.time_cost,
                                time.time(),
                            )

                        else:
                            if "InstantiationError" in str(exception):
                                msg = str(exception)
                                try:
                                    msg = msg.split("\n")[-2].split(": ")[1]
                                except Exception:  # pylint: disable=broad-except
                                    pass
                                res = MeasureResult(
                                    (
                                        tb,
                                        InstantiationError(msg),
                                    ),
                                    MeasureErrorNo.INSTANTIATION_ERROR,
                                    res.time_cost,
                                    time.time(),
                                )

                            else:  # tvm error
                                res = MeasureResult(
                                    (
                                        tb,
                                        res.error,
                                    ),
                                    MeasureErrorNo.COMPILE_HOST,
                                    res.time_cost,
                                    time.time(),
                                )
                except TimeoutError as ex:
                    tb = traceback.format_exc()
                    res = MeasureResult(
                        (
                            tb,
                            ex,
                        ),
                        MeasureErrorNo.BUILD_TIMEOUT,
                        self.timeout,
                        time.time(),
                    )
                except ChildProcessError as ex:
                    tb = traceback.format_exc()
                    res = MeasureResult(
                        (
                            tb,
                            ex,
                        ),
                        MeasureErrorNo.RUNTIME_DEVICE,
                        self.timeout,
                        time.time(),
                    )

                results.append(res)

        return results


class RPCRunner(Runner):
    """Run generated code on remove devices.
    This function will ask a RPC Tracker to get device for measurement.

    Parameters
    ----------
    timeout: float
        The timeout of a RPCRunner measurement task
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    key: str
        The key of the device registered in the tracker
    host: str
        The host address of RPC Tracker
    port: int
        The port of RPC Tracker
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    module_loader : ModuleLoader
        If given, a context manager that loads the module to be timed into the remote runtime.
        If not given, default_module_loader is used.
    """

    def __init__(
        self,
        key,
        host,
        port,
        priority=1,
        timeout=10,
        n_parallel=None,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        enable_cpu_cache_flush=False,
        module_loader=None,
    ):
        super(RPCRunner, self).__init__(timeout, n_parallel)

        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
        self.timeout = timeout

        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms
        self._ref_input = None

        self.enable_cpu_cache_flush = enable_cpu_cache_flush
        self.cooldown_interval = cooldown_interval
        self.module_loader = module_loader

        do_fork = True
        if do_fork:
            self.executor = PopenPoolExecutor(
                max_workers=1,
                timeout=timeout * (self.n_parallel),
                initializer=reset_global_scope,
                initargs=(AutotvmGlobalScope.current,),
            )
        else:
            self.executor = None

    @property
    def ref_input(self):
        """
        Fixed input for tuning special operators, e.g., sparse operators
        requiring indices as input.
        """
        return self._ref_input

    @ref_input.setter
    def ref_input(self, val):
        if val is not None:
            warnings.warn(
                "You are specifying fixed input for tuning the operator. "
                "Be sure your input always fits the operator. Some "
                "operators may conduct layout transformation during tuning, "
                "thus can lead to unexpected behaviors. ",
                RuntimeWarning,
            )
        self._ref_input = val

    def set_task(self, task):
        self.task = task

        # BUG(fucheng): cuda and webgpu cannot be checked.
        if task.target.kind.name == "webgpu":
            return
        
        if check_remote(task.target, self.key, self.host, self.port):
            logger.info("Get devices for measurement successfully!")
        else:
            raise RuntimeError(
                "Cannot get remote devices from the tracker. "
                "Please check the status of tracker by "
                "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                "and make sure you have free devices on the queue status."
            )

    def get_build_kwargs(self):
        kwargs = {}
        if (
            "cuda" in self.task.target.keys
            or "opencl" in self.task.target.keys
            or "rocm" in self.task.target.keys
            or "vulkan" in self.task.target.keys
        ):
            # NOTE(fucheng): We set constrains for sm_86 here.
            if self.task.target.arch == "sm_86":
                max_dims = [1024, 1024, 1024]
                kwargs["check_gpu"] = {
                    "max_shared_memory_per_block": 49152,
                    "max_threads_per_block": 1024,
                    "max_thread_x": max_dims[0],
                    "max_thread_y": max_dims[1],
                    "max_thread_z": max_dims[2],
                }
            else:
                remote = request_remote(self.key, self.host, self.port)
                dev = remote.device(str(self.task.target), 0)
                max_dims = dev.max_thread_dimensions
                kwargs["check_gpu"] = {
                    "max_shared_memory_per_block": dev.max_shared_memory_per_block,
                    "max_threads_per_block": dev.max_threads_per_block,
                    "max_thread_x": max_dims[0],
                    "max_thread_y": max_dims[1],
                    "max_thread_z": max_dims[2],
                }
            print("measure_methods.py: check_gpu", kwargs["check_gpu"])
        elif (self.task.target.kind.name == "webgpu"):
            # NOTE(fucheng): We set constrains for webgpu here.
            max_dims = [256, 256, 64]
            kwargs["check_gpu"] = {
                "max_shared_memory_per_block": 16384,
                "max_threads_per_block": 256,
                "max_thread_x": max_dims[0],
                "max_thread_y": max_dims[1],
                "max_thread_z": max_dims[2],
            }

        return kwargs

    def run(self, measure_inputs, build_results):
        #print("measure_methods.py: Call RPCRunner.run")
        #print("measure_methods.py: self.module_loader", self.module_loader)
        #print("measure_methods.py: measure_inputs", measure_inputs)
        #print("measure_methods.py: build_results", build_results)
        results = []
        remote_kwargs = dict(
            device_key=self.key,
            host=self.host,
            port=self.port,
            priority=self.priority,
            timeout=self.timeout,
        )

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(
                measure_inputs[i : i + self.n_parallel], build_results[i : i + self.n_parallel]
            ):
                global browser_server
                browser_pid = None
                if not isinstance(build_res, MeasureResult):
                    if browser_server is not None:
                        browser_pid = browser_server.open_chrome()
                        time.sleep(5)

                module_loader = (
                    self.module_loader
                    if self.module_loader is not None
                    else default_module_loader()
                )
                if self.executor is not None:
                    ret = self.executor.submit(
                        run_through_rpc,
                        measure_inp,
                        build_res,
                        self.number,
                        self.repeat,
                        self.min_repeat_ms,
                        self.cooldown_interval,
                        remote_kwargs,
                        self.ref_input,
                        self.enable_cpu_cache_flush,
                        module_loader,
                    )
                else:
                    ret = run_through_rpc(
                        measure_inp,
                        build_res,
                        self.number,
                        self.repeat,
                        self.min_repeat_ms,
                        self.cooldown_interval,
                        remote_kwargs,
                        self.ref_input,
                        self.enable_cpu_cache_flush,
                        module_loader,
                    )
                futures.append(ret)

            for future in futures:
                try:
                    if self.executor is not None:
                        res = future.result()
                    else:
                        res = future
                    results.append(res)
                except Exception as ex:  # pylint: disable=broad-except
                    tb = traceback.format_exc()
                    results.append(
                        MeasureResult(
                            (
                                tb,
                                ex,
                            ),
                            MeasureErrorNo.RUN_TIMEOUT,
                            self.timeout,
                            time.time(),
                        )
                    )
                if browser_server is not None and browser_pid is not None:
                    #input("measure_methods.py: Press ENTER to continue")
                    browser_server.close_chrome(browser_pid)

        return results


class LocalRunner(RPCRunner):
    """Run generated code on local devices.

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    Note
    ----
    This is a "fake" local mode. We start a silent rpc tracker and rpc server
    for the user. In this way we reuse timeout/isolation mechanism in RPC infrastructure.
    """

    def __init__(
        self,
        timeout=10,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        enable_cpu_cache_flush=False,
        module_loader=None,
    ):
        super(LocalRunner, self).__init__(
            "",
            None,
            None,
            0,
            timeout=timeout,
            n_parallel=1,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            cooldown_interval=cooldown_interval,
            enable_cpu_cache_flush=enable_cpu_cache_flush,
            module_loader=module_loader,
        )
        self.tracker = None
        self.server = None

    def set_task(self, task):
        # pylint: disable=import-outside-toplevel
        from ...rpc.server import Server
        from ...rpc.tracker import Tracker

        self.task = task
        tracker = Tracker(port=9000, port_end=10000, silent=True)
        device_key = "$local$device$%d" % tracker.port
        server = Server(
            port=9000,
            port_end=10000,
            key=device_key,
            silent=True,
            tracker_addr=("127.0.0.1", tracker.port),
        )
        self.key = device_key
        self.host = "127.0.0.1"
        self.port = tracker.port

        super(LocalRunner, self).set_task(task)
        return server, tracker


def _build_func_common(measure_input, runtime=None, check_gpu=None, build_option=None):
    """Common part for building a configuration"""
    target, task, config = measure_input
    
    #print("measure_methods.py: target:", target)
    #print("measure_methods.py: task:", task)
    #print("measure_methods.py: config:", config)
    #print("measure_methods.py: flatten_feature:", config.get_flatten_feature())
    #input("measure_methods.py: paused")
    target_config_idx = None
    if target_config_idx is not None:
        #print("measure_methods.py: target_config_idx %d" % target_config_idx)
        if config.index != target_config_idx:
            raise ValueError("Not specified config index")
    
    from .verify import verify_kernel_hw_occu
    enable_hw_verification = get_os_env_var_bool("TVM_ENABLE_HW_VERYFICATION", False)
    #enable_hw_verification = False
    if enable_hw_verification and (config is not None) and \
            not verify_kernel_hw_occu(target.kind.name, task, config.get_flatten_feature()):
        raise ValueError("Not verified kernel hardware occupancy")
    #input("measure_methods.py: paused")
    
    if config is not None and False:
        flatten_config = config.get_flatten_feature()
        #print("measure_methods.py: flatten_config:", flatten_config)
        #num_vthreads_x = int(flatten_config[5])
        #num_vthreads_y = int(flatten_config[9])
        #num_threads_x = int(flatten_config[6])
        #num_threads_y = int(flatten_config[10])
        #num_vthreads = num_vthreads_x * num_vthreads_y
        #num_threads = num_threads_x * num_threads_y
        #if num_vthreads_x == 1 or num_vthreads_y == 1:
        #    raise ValueError("Bad virtual thread number")
        #if num_threads_x == 1 or num_threads_y == 1:
        #    raise ValueError("Bad thread number")
        #if num_vthreads > 1 and num_threads > 1:
        #    raise ValueError("Bad thread number")
        #if (num_threads_x * num_threads_y) % 32 > 0:
        #    raise ValueError("Bank conflict")
    #print("measure_methods.py: verified kernel hardware occupancy")
    #input("measure_methods.py: paused")
    
    target, task.target_host = Target.canon_target_and_host(target, task.target_host)

    with target:
        s, args = task.instantiate(config)
        if get_os_env_var_bool("TVM_ENABLE_TIR_LOG", False):
            print("measure_methods.py: config " + str(config))
            print("measure_methods.py: TVM IR")
            print(tvm.lower(s, args, simple_mode=False))
            #input("measure_methods.py: paused")
            #exit(0)

        # check invalidity of template and code hash consistency
        if config is not None and not config.valid():
            raise InstantiationError(config.errors)

        # if target is vta, we need to use vta build
        if (
            hasattr(measure_input.target, "device_name")
            and measure_input.target.device_name == "vta"
        ):
            # pylint: disable=import-outside-toplevel
            import vta

            func = vta.build(s, args, target_host=task.target_host)
        else:
            current_pass_context: tvm.ir.transform.PassContext = (
                tvm.ir.transform.PassContext.current()
            )
            current_config = dict(current_pass_context.config)
            if build_option is not None:
                current_config.update(build_option)

            if "tir.add_lower_pass" in current_config:
                current_add_lower_pass = list(current_config["tir.add_lower_pass"])
            else:
                current_add_lower_pass = []
            if check_gpu:
                current_add_lower_pass.append((2, gpu_verify_pass(**check_gpu)))
            current_config["tir.add_lower_pass"] = current_add_lower_pass

            #print("measure_methods.py: current_pass_context %s, current_config %s" % (
            #        current_pass_context, current_config))

            with tvm.ir.transform.PassContext(
                opt_level=current_pass_context.opt_level,
                required_pass=current_pass_context.required_pass,
                disabled_pass=current_pass_context.disabled_pass,
                instruments=current_pass_context.instruments,
                config=current_config,
            ):
                func = build(s, args, target_host=task.target_host, runtime=runtime)
    return func, tuple((get_const_tuple(x.shape), x.dtype) for x in args)


class _WrappedBuildFunc:
    """
    Wrap build_func to a function that can be used in measure.

    Note: this is a class instead of a closure so that it can be pickled when
    using multiprocessing.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format".
    runtime : Optional[Runtime]
        The runtime to generate artifacts for

    Returns
    -------
    wrapped_build_func : callable
        The wrapped build function
    """

    def __init__(self, build_func, runtime=None):
        if not hasattr(build_func, "output_format"):
            raise AttributeError("Expect build_func to have the attribute output_format.")
        self.build_func = build_func
        self.runtime = runtime

    def __call__(self, measure_input, tmp_dir, **kwargs):
        """
        Wrapped build func.

        Parameters
        ----------
        measure_input: MeasureInput
            The input of measurement

        tmp_dir: str
            The path of temporary directory to export generated library
        """
        tic = time.time()
        try:
            filename = os.path.join(
                tmp_dir, "tmp_func_%0x.%s" % (getrandbits(64), self.build_func.output_format)
            )
            if MEASURE_METHODS_CONFIGS["enable_build"]:
                global total_kernel_count, built_kernel_count
                if total_kernel_count is not None:
                    total_kernel_count += 1
                # TODO(tvm-team) consider linline _build_func_common
                func, arg_info = _build_func_common(measure_input, self.runtime, **kwargs)
                if get_os_env_var_bool("TVM_ENABLE_BUILD_LOG", False):
                    print("measure_methods.py: build_func_time_cost " + str(time.time() - tic))
                if MEASURE_METHODS_CONFIGS["enable_export"]:
                    if get_os_env_var_bool("TVM_ENABLE_BUILD_LOG", False):
                        print("measure_methods.py: exported_file " + filename)
                    if self.build_func.output_format == ".model-library-format":
                        # Late import to preserve autoTVM with USE_MICRO OFF
                        try:
                            from tvm import micro  # pylint: disable=import-outside-toplevel
                        except ImportError:
                            raise ImportError("Requires USE_MICRO")
                        micro.export_model_library_format(func, filename)
                    else:
                        func.export_library(filename, self.build_func)  
                    if get_os_env_var_bool("TVM_ENABLE_BUILD_LOG", False):
                        print("measure_methods.py: export file done")
                if built_kernel_count is not None:
                    built_kernel_count += 1
                    print("measure_methods.py: built_kernel_count %d/%d" % (built_kernel_count, total_kernel_count))
        except Exception as e:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            return BuildResult(None, None, (tb, e), time.time() - tic)
        if get_os_env_var_bool("TVM_ENABLE_BUILD_LOG", False):
            print("measure_methods.py: build_time_cost " + str(time.time() - tic))
        return BuildResult(filename, arg_info, None, time.time() - tic)


ModuleLoader = typing.Callable[
    [dict, dict], typing.ContextManager[typing.Tuple[tvm.rpc.RPCSession, tvm.runtime.Module]]
]


def run_through_rpc(
    measure_input,
    build_result,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    remote_kwargs,
    ref_input,
    enable_cpu_cache_flush=False,
    module_loader=None,
):
    """Run a generated library through rpc

    Parameters
    ----------
    measure_input: MeasureInput
        The raw measure input
    build_result: BuildResult
        The result returned from Builder. This contains the path to the generated library.
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float
        The cool down interval between two measurements
    remote_kwargs: dict
        Passed to module_loader(). Ultimately, keyword args to request_remote().
    ref_input: List of np.ndarray
        The reference input used for tuning. Empty for randomly filled input.
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    module_loader: ModuleLoader
        A function that returns a ContextManager used to establish and teardown the remote session.
    """
    #print("measure_methods.py: Call run_through_rpc")
    enable_run = MEASURE_METHODS_CONFIGS["enable_run"]
    if not enable_run:
        costs = (1e9)
        errno = MeasureErrorNo.NO_ERROR
        tstamp = tic = time.time()
        return MeasureResult(costs, errno, tstamp - tic, tstamp)
    
    if isinstance(build_result, MeasureResult):
        #print("measure_methods.py: Return built result")
        return build_result

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR
    try:
        # upload built module
        with module_loader(remote_kwargs, build_result) as (remote, mod):
            if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                print("measure_methods.py: target %s, target_kind %s, target_keys %s" % (
                        measure_input.target, measure_input.target.kind, measure_input.target.keys))
            
            if "wasm" in str(measure_input.target) or measure_input.target.kind.name == "webgpu":
                #print("measure_methods.py: Upload built module")
                #dev = remote.cpu(0)
                dev = remote.device(str(measure_input.target), 0)
                #print("measure_methods.py: measure_input.target %s, dev %s" % (str(measure_input.target), dev))

                if ref_input:
                    args = [nd.array(x, device=dev) for x in ref_input]
                else:
                    args = [nd.empty(x[0], x[1], dev) for x in build_result.arg_info]
                    dev.sync()

                fname = "default_function"
                
                if measure_input.target.kind.name == "webgpu":
                    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                        print("measure_methods.py: webgpu")
                    
                    #print("measure_methods.py: register webgpu kernel", fname)
                    #fregisterwebgpukernelfunc = remote.get_function("__sync.wasm.registerWebGPUKernelFunc")
                    ftimeexec = remote.get_function("__sync.wasm.TimeExecutionForWebGPU")
                    fiswebgpufinished = remote.get_function("__sync.wasm.isTimeExecutionForWebGPUFinished")
                    fgetwebgpuresults = remote.get_function("__sync.wasm.getTimeExecutionForWebGPUResults")
                    
                    #fregisterwebgpukernelfunc(fname)
                    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                        print("measure_methods.py: executing...")
                    ftimeexec(fname, dev, number, *args)
                    while fiswebgpufinished() == 0:
                        time.sleep(1)
                    cost_bytes = fgetwebgpuresults()
                else:
                    if measure_input.target.kind.name == "llvm":
                        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                            print("measure_methods.py: llvm-wasm")
                    else:
                        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                            print("measure_methods.py: wasm")
                        #fname = "addone"
                    
                    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                        print("measure_methods.py: executing...")
                    ftimeexec = remote.get_function("__sync.wasm.TimeExecutionForWasm")
                    cost_bytes = ftimeexec(fname,
                                           dev,
                                           number,
                                           repeat,
                                           min_repeat_ms,
                                           10,
                                           cooldown_interval,
                                           10,
                                           *args)

                import numpy as np
                #cost_arr = np.frombuffer(cost_bytes, dtype=np.float64)
                cost_arr = np.frombuffer(cost_bytes, dtype=np.dtype("<f8"))
                costs = tuple(cost_arr.tolist())
                #print("measure_methods.py: costs", costs)
            elif measure_input.target.kind.name == "cuda":
                if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                    print("measure_methods.py: cuda")
                enable_not_run_cuda = False
                if enable_not_run_cuda:
                    costs = [1e9]
                else:
                    dev = remote.device(str(measure_input.target), 0)
                    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                        print("measure_methods.py: dev", dev)
                    time_f = mod.time_evaluator(
                        mod.entry_name,
                        dev,
                        number=1,
                        repeat=50,
                        min_repeat_ms=100
                    )
                    
                    if ref_input:
                        args = [nd.array(x, device=dev) for x in ref_input]
                    else:
                        try:
                            random_fill = remote.get_function("tvm.contrib.random.random_fill")
                        except AttributeError:
                            raise AttributeError(
                                "Please make sure USE_RANDOM is ON in the config.cmake "
                                "on the remote devices"
                            )
                        args = [nd.empty(x[0], x[1], dev) for x in build_result.arg_info]
                        if "scatter" not in measure_input.task.name:
                            # the index tensor of scatter op cannot be randomly initialized
                            for arg in args:
                                random_fill(arg)
                        dev.sync()

                    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                        print("measure_methods.py: executing...")
                    #mod(*args)

                    #time_f(*args)
                    #costs = [1e9]
                    costs = time_f(*args).results
            else:
                if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
                    print("measure_methods.py: other")

                dev = remote.device(str(measure_input.target), 0)
                # Limitation:
                # We can not get PackFunction directly in the remote mode as it is wrapped
                # under the std::function. We could lift the restriction later once we fold
                # the PackedFunc as an object. Currently, we pass function name to work
                # around it.
                f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
                time_f = mod.time_evaluator(
                    mod.entry_name,
                    dev,
                    number=number,
                    repeat=repeat,
                    min_repeat_ms=min_repeat_ms,
                    f_preproc=f_prepare,
                )

                if ref_input:
                    args = [nd.array(x, device=dev) for x in ref_input]
                else:
                    try:
                        random_fill = remote.get_function("tvm.contrib.random.random_fill")
                    except AttributeError:
                        raise AttributeError(
                            "Please make sure USE_RANDOM is ON in the config.cmake "
                            "on the remote devices"
                        )
                    args = [nd.empty(x[0], x[1], dev) for x in build_result.arg_info]
                    if "scatter" not in measure_input.task.name:
                        # the index tensor of scatter op cannot be randomly initialized
                        for arg in args:
                            random_fill(arg)
                    dev.sync()

                costs = time_f(*args).results

        if len(costs) > 2:  # remove largest and smallest value to reduce variance
            costs = list(costs)
            costs.sort()
            costs = tuple(costs[1:-1])

        if np.mean(np.array(costs)) == 0.0:
            costs = (1e9)

        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
            print("measure_methods.py: costs", costs)
            print("measure_methods.py: avg_cost", np.mean(np.array(costs)))
            print("measure_methods.py: median_cost", np.median(np.array(costs)))
    except TVMError as exc:
        msg = str(exc)
        if "Stack trace returned" in msg:
            msg = msg[: msg.index("Stack trace returned")]
        if "CUDA Source" in msg:
            msg = msg[: msg.index("CUDA Source")]
        costs = (
            traceback.format_exc(),
            RuntimeError(msg[:1024]),
        )
        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)
    #print("measure_methods.py: filename %s, t_time_evaluation %f" % (build_result.filename, (tstamp - tic)))
    return MeasureResult(costs, errno, tstamp - tic + build_result.time_cost, tstamp)


class DefaultModuleLoader:
    """See default_module_loader(). A pickleable emulation of the original function closure."""

    def __init__(self, pre_load_function=None) -> None:
        self.pre_load_function = pre_load_function

    @contextlib.contextmanager
    def __call__(self, remote_kwargs, build_result):
        remote = request_remote(**remote_kwargs)
        if self.pre_load_function is not None:
            self.pre_load_function(remote, build_result)

        remote.upload(build_result.filename)
        try:
            yield remote, remote.load_module(os.path.split(build_result.filename)[1])

        finally:
            # clean up remote files
            remote.remove(build_result.filename)
            remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
            remote.remove("")


class WasmModuleLoader:
    def __init__(self, pre_load_function=None) -> None:
        #print("measure_methods.py: Create WasmModuleLoader")
        self.pre_load_function = pre_load_function

    @contextlib.contextmanager
    def __call__(self, remote_kwargs, build_result):
        #print("measure_methods.py: Call WasmModuleLoader")
        if self.pre_load_function is not None:
            self.pre_load_function(remote, build_result)

        wasm_path = build_result.filename
        #print("measure_methods.py: Run " + wasm_path)

        wasm_binary = open(wasm_path, "rb").read()
        from tvm import rpc
        proxy_host = "127.0.0.1"
        proxy_port = os.getenv("PROXY_PORT")
        proxy_port = 9090 if proxy_port is None else int(proxy_port)
        key = os.getenv("PROXY_KEY")
        key = "wasm" if key is None else key
        session_timeout = 10
        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
            print("measure_methods.py: RPC connect, host %s, port %d, key %s, timeout %d" % (
                proxy_host, proxy_port, key, session_timeout
            ))
        remote = rpc.connect(
            proxy_host,
            proxy_port,
            key=key,
            session_timeout=session_timeout,
            session_constructor_args=["rpc.WasmSession", wasm_binary],
        )
        yield remote, remote


def default_module_loader(pre_load_function=None):
    """Returns a default function that can be passed as module_loader to run_through_rpc.

    Parameters
    ----------
    pre_load_function : Optional[Function[tvm.rpc.Session, tvm.runtime.Module]]
        Invoked after a session is established and before the default code-loading RPC calls are
        issued. Allows performing pre-upload actions, e.g. resetting the remote runtime environment.

    Returns
    -------
    DefaultModuleLoader :
        A callable that can be passed as module_loader to run_through_rpc.
    """

    # This was a function with a closure before but that couldn't be pickled!
    # We need pickle to work for using python's multiprocessing on some platforms.
    return DefaultModuleLoader(pre_load_function)


def wasm_module_loader(pre_load_function=None):
    return WasmModuleLoader(pre_load_function)


def request_remote(device_key, host=None, port=None, priority=1, timeout=60):
    """Request a remote session

    Parameters
    ----------
    device_key: string
        The device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this session (units: second)

    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])

    tracker = _rpc.connect_tracker(host, port)
    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
        print("measure_methods.py: connected tracker, host %s, port %d" % (host, port))
    #timeout = 60
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
        print("measure_methods.py: tracker required, device_key %s, timeout %s" % (device_key, timeout))
    return remote


def check_remote(target, device_key, host=None, port=None, priority=100, timeout=10):
    """
    Check the availability of a remote device

    Parameters
    ----------
    target: Target
        The wanted compilation target
    device_key: string
        device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this check (units: seconds).

    Returns
    -------
    available: bool
        True if can find available device
    """

    def _check():
        logger.debug("waiting for device...")
        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
            print("measure_methods.py: waiting for device...")
            print("measure_methods.py: host %s, port %d, device_key %s" % (host, port, device_key))
        remote = request_remote(device_key, host, port, priority)
        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
            print("measure_methods.py: remote available")
        dev = remote.device(str(target))
        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
            print("measure_methods.py: device available")
        while not dev.exist and target.kind.name != "cuda":  # wait until we get an available device
            pass
        if get_os_env_var_bool("TVM_ENABLE_RUN_LOG", False):
            print("measure_methods.py: device exist")
        logger.debug("device available")

    t = threading.Thread(
        target=_check,
    )
    t.start()
    t.join(timeout)

    remote = request_remote(device_key, host, port, priority)
    dev = remote.device(str(target))
    if target.kind.name == "cuda":
        return True
    else:
        return dev.exist


def set_cuda_target_arch(arch):
    """THIS API IS DEPRECATED.

    set target architecture of nvcc compiler

    Parameters
    ----------
    arch: str or list
        The argument of nvcc -arch. (e.g. "sm_51", "sm_62")
        it can also be a count of gencode arguments pass to nvcc command line,
        e.g., ["-gencode", "arch=compute_52,code=sm_52", "-gencode", "arch=compute_70,code=sm_70"]
    """
    raise ValueError(
        "The API 'autotvm.measure.set_cuda_target_arch' is deprecated."
        "Try specifying it by adding '-arch=sm_xx' to your target, such as 'cuda -arch=sm_86'."
        "See https://github.com/apache/tvm/pull/9544 for the upgrade guide."
    )


def gpu_verify_pass(**kwargs):
    """Verify the validity of a gpu kernel.
    This pass will check memory usage and number of threads per block.
    """

    def verify_pass(f, *_):
        valid = tvm.tir.analysis.verify_gpu_code(f, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return f

    return tvm.tir.transform.prim_func_pass(verify_pass, opt_level=0)
