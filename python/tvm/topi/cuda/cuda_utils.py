
import tvm


def is_webgpu_cuda():
    target = tvm.target.Target.current(allow_none=False)
    print("target", target)
    return "webgpu" in target.keys
