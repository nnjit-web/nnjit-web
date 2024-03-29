
import tvm
from tvm import te
import numpy as np
from .kernel import Kernel


class VecFMA(Kernel):
    def __init__(self, L):
        super().__init__()
        self._L = L
        self._mod_name = "vecfma"
        self._flops = 4 * L
        self._target = None


    def build(self, target, runtime):
        self._target = target
        
        #n = te.var("n")
        n = self._L
        #l = te.reduce_axis((0, self._L), "l")
        self._A = A = te.placeholder((n,), name="A")
        B = te.placeholder((n,), name="B")
        #C = te.placeholder((n,), name="C")
        #D = te.compute((n,), lambda l: A(l // 32768) * B(l // 32768) + A(l // 32768), name="D")
        
        C = te.compute((n,), lambda l: A(l) * B(l) + A(l), name="C")
        D = te.compute((n,), lambda l: C(l // 32768) * A(l // 32768) + C(l // 32768), name="D")

        s = te.create_schedule(D.op)

        if target.kind.name == "llvm":
            enable_vectorization = False
            if enable_vectorization:
                lo, li = s[D].split(D.op.axis[0], factor=4)
                s[D].reorder(lo, li)
                s[D].vectorize(li)
        elif target.kind.name == "webgpu":
            num_thread = 64
            #CC = s.cache_write(C, "local")
            #CD = s.cache_write(D, "local")
            xo, xi = s[D].split(D.op.axis[0], factor=512)
            xoo, xoi = s[D].split(xo, factor=num_thread)
            s[C].compute_at(s[D], xi)
            s[D].bind(xoo, te.thread_axis("blockIdx.x"))
            s[D].bind(xoi, te.thread_axis("threadIdx.x"))

            #s[CC].compute_at(s[D], xi)
            #s[CD].compute_at(s[D], xi)
        else:
            raise ValueError("Unsupported target: " + target.kind.name)

        mod = tvm.build(s, [A, B, D], target, runtime=runtime, name=self._mod_name)
        print(tvm.lower(s, [A, B, D], simple_mode=True))
        return mod


    def test_rpc(self, remote_system_lib, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=self._L).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=self._L).astype(dtype), dev)
        c = tvm.nd.array(np.zeros(self._L, dtype=dtype), dev)
        d = tvm.nd.array(np.zeros(self._L, dtype=dtype), dev)
        # invoke the function
        remote_func = remote_system_lib.get_function(self._mod_name)
        remote_func(a, b, c, d)
        np.testing.assert_equal(d.numpy(), a.numpy() * b.numpy() + c.numpy())
        print("Test pass!")


    def test_time_evaluation(self, remote_system_lib, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=self._L).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=self._L).astype(dtype), dev)
        c = tvm.nd.array(np.zeros(self._L, dtype=dtype), dev)
        d = tvm.nd.array(np.zeros(self._L, dtype=dtype), dev)

        if self._target.kind.name == "llvm":
            time_f = remote_system_lib.get_function("__sync.wasm.TimeExecutionForWasm")
            nstep = 100
            repeat = 10
            cost_bytes = time_f(self._mod_name, dev, nstep, repeat, 0, 0, 0, 0, a, b, c, d)
        elif self._target.kind.name == "webgpu":
            time_f = remote_system_lib.get_function("__sync.wasm.TimeExecutionForWebGPU")
            is_finished_f = remote_system_lib.get_function("__sync.wasm.isTimeExecutionForWebGPUFinished")
            get_ret_f = remote_system_lib.get_function("__sync.wasm.getTimeExecutionForWebGPUResults")

            repeat = 64
            time_f(self._mod_name, dev, repeat, a, b, c, d)
            while is_finished_f() == 0:
                import time
                time.sleep(1)
            cost_bytes = get_ret_f()
        else:
            raise ValueError("Unsupported target: " + self._target.kind.name)
        
        cost_arr = np.frombuffer(cost_bytes, dtype=np.dtype("<f8"))
        print(cost_arr)
        cost = np.mean(cost_arr)
        print("%g secs/op" % cost)
        gflops = self._flops * 1e-9 / cost
        print("%f GFLOPS/op" % gflops)
