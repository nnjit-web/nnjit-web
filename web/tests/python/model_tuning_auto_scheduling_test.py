
import os
import numpy as np

import argparse
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.relay.backend import Runtime


parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, required=True, help="Network")
parser.add_argument("--dev-info", type=str, required=True, help="Device info")
parser.add_argument("--num-trials", type=int, default=1000, help="Trials number")
args = parser.parse_args()


def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    print("layout:", layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name == "alexnet":
        mod, params = relay.testing.alexnet.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape
        )
    elif name == "vgg":
        mod, params = relay.testing.vgg.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape
        )
    elif name == "densenet":
        mod, params = relay.testing.densenet.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape
        )
    elif name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target.
# If the target machine supports avx512 instructions, replace the
# "llvm -mcpu=core-avx2" with "llvm -mcpu=skylake-avx512"
network = args.network
use_sparse = False
batch_size = 1
layout = "NCHW"
dtype = "float32"
use_conv_gemm = os.getenv("TVM_WASM_GEMM_CONV")
use_conv_gemm = (use_conv_gemm is not None) and (use_conv_gemm == "1")
if use_conv_gemm:
    layout = "NHWC"
    dtype = "int8"
runtime = Runtime("cpp", {"system-lib": True})
target = "llvm -mtriple=wasm32-unknown-unknown-wasm"
if use_conv_gemm:
    target = target + " -keys=arm_cpu"
use_simd = os.getenv("EMCC_USE_SIMD")
use_simd = (use_simd is not None) and (use_simd == "1")
if use_simd:
    target = target + " -mattr=+simd128"
target_kind_name = "llvm"
if not tvm.runtime.enabled(target):
    raise RuntimeError("Target %s is not enbaled" % target)
opt_level = os.getenv("EMCC_OPT_LEVEL")
opt_level = "-O3" if opt_level is None else opt_level
print("opt_level %s, use_simd %s" % (opt_level, use_simd))

dev_info = args.dev_info
compiler_info = opt_level[1:].lower()
simd_info = "simd" if use_simd else "nosimd"
conv_info = "gemmconv" if use_conv_gemm else "dconv"
log_file = "%s-%s-B%d-%s-%s-%s-%s-%s.json" % (network, layout, batch_size, target_kind_name, dev_info, compiler_info, simd_info, conv_info)

# Extract tasks from the network
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network,
    batch_size,
    layout,
    dtype=dtype,
    use_sparse=use_sparse,
)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

task_flop_dict = {}
for idx, task in enumerate(tasks):
    print("task_key %s, FLOP %d" % (task.workload_key, task.compute_dag.flop_ct))
    task_flop_dict[task.workload_key] = task.compute_dag.flop_ct

def run_tuning():
    print("Begin tuning...")
    timeout = 60
    number = 50
    repeat = 1
    num_measure_trials = 10000
    print("timeout %d, repeat %d, num_measure_trials %d" % (timeout, repeat, num_measure_trials))
    builder = auto_scheduler.LocalBuilder(timeout=timeout,
                                          n_parallel=1,
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


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

run_tuning_flag = True

if run_tuning_flag:
  run_tuning()
else:
  print("Skip tuning")

# Compile with the history best
print("Compile...")
print("Load log...")
with auto_scheduler.ApplyHistoryBest(log_file, task_flop_dict=task_flop_dict):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
