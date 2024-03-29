
import time
import tvm
from tvm import te
import numpy as np
from .kernel import Kernel


class MatMul(Kernel):
    def __init__(self, M, K, N, target):
        super().__init__()
        self._mod_name = "matmul"
        self._M = M
        self._K = K
        self._N = N
        self._target = target
        self._a_size = (self._M, self._K)
        self._b_size = (self._K, self._N)
        self._c_size = (self._M, self._N)

        self._enable_cache_write = False
        self._enable_packing = False
        self._enable_parallel = True

    
    def numpy_compute(self, a, b, c):
        for i in range(0, self._M):
            for j in range(0, self._N):
                for k in range(0, self._K):
                    c[i, j] += a[i, k] * b[k, j]


    def add_schedule(self):
        C = self._C
        s = self._s

        mbs = 4
        nbs = 4
        kbs = 32

        if not self._enable_packing:
            if not self._enable_cache_write:
                m, n = C.op.axis
                (k,) = C.op.reduce_axis
                mo, no, mi, ni = s[C].tile(m, n, mbs, nbs)
                ko, ki = s[C].split(k, kbs)
                s[C].reorder(mo, no, ko, ki, mi, ni)
                s[C].unroll(mi)
                s[C].vectorize(ni)
                if self._enable_parallel:
                    s[C].parallel(mo)
            else:
                CC = s.cache_write(C, "global")

                m, n = C.op.axis
                #(k,) = C.op.reduce_axis

                mo, no, mi, ni = s[C].tile(m, n, mbs, nbs)
                s[C].unroll(mi)
                s[C].vectorize(ni)
                
                s[CC].compute_at(s[C], no)

                mc, nc = CC.op.axis
                (k,) = CC.op.reduce_axis
                ko, ki = s[CC].split(k, kbs)

                s[CC].reorder(ko, ki, mc, nc)

                #s[CC].unroll(ni)
                
                s[CC].unroll(mc)
                s[CC].vectorize(nc)
        else:
            m, n = C.op.axis
            (k,) = C.op.reduce_axis

            mo, no, mi, ni = s[C].tile(m, n, mbs, nbs)
            ko, ki = s[C].split(k, kbs)
            s[C].reorder(mo, no, ko, ki, mi, ni)


    def build(self, target, runtime):
        k = te.reduce_axis((0, self._K), "k")
        self._A = A = te.placeholder(self._a_size, name="A")
        self._B = B = te.placeholder(self._b_size, name="B")
        if not self._enable_packing:
            self._C = C = te.compute(
                    self._c_size,
                    lambda m, n: te.sum(A[m, k] * B[k, n], axis=k),
                    name="C"
            )
            args = [A, B, C]
        else:
            bn = 4
            self._packed_b_size = packed_b_size = (self._N // bn, self._K, bn)
            self._packed_B = packed_B = te.compute(
                    packed_b_size,
                    lambda big_n, k, little_n: B[k, big_n * bn + little_n],
                    name="packedB"
            )
            self._C = C = te.compute(
                    self._c_size,
                    lambda m, n: te.sum(A[m, k] * packed_B[n // bn, k, n % bn], axis=k),
                    name="C"
            )
            args = [A, B, C]

        self._s = s = te.create_schedule(C.op)
        self.add_schedule()

        print(tvm.lower(s, args, simple_mode=True))
        time.sleep(3)

        if self._target == "llvm":
            self._func_name = self._mod_name
            mod_name = self._mod_name
        elif self._target == "wasm":
            self._func_name = "addone"
            func_code_name = "1599"
            mod_name = func_code_name
        
        mod = tvm.build(s, args, target, runtime=runtime, name=mod_name)
        return mod


    def test_rpc(self, remote_system_lib, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=self._a_size).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=self._b_size).astype(dtype), dev)
        out = tvm.nd.array(np.zeros(self._c_size, dtype=dtype), dev)
        matmul_func = remote_system_lib.get_function(self._func_name)
        matmul_func(a, b, out)
        print("a:", a.numpy())
        print("b:", b.numpy())
        print("out:", out.numpy())
        out_numpy = np.zeros(self._c_size, dtype=dtype)
        self.numpy_compute(a.numpy(), b.numpy(), out_numpy)
        print("out_numpy:", out_numpy)
        np.testing.assert_equal(out.numpy(), out_numpy)
        print("Test pass!")


    def test_time_evaluation(self, remote, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=self._a_size).astype(dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=self._b_size).astype(dtype), dev)
        out = tvm.nd.array(np.zeros(self._c_size, dtype=dtype), dev)
        '''
        time_f = remote.system_lib().time_evaluator(self._func_name, dev, number=100, repeat=10)
        cost = time_f(a, b, out).mean
        '''
        time_f = remote.get_function("__sync.wasm.TimeExecutionForWasm")
        nstep = 1
        repeat = 10
        cost_bytes = time_f(self._func_name, dev, nstep, repeat, 0, 0, 0, 0, a, b, out)
        costs = np.frombuffer(cost_bytes, dtype=np.dtype("<f8"))
        cost = np.mean(costs)
        gflops = 2 * self._M * self._K * self._N * 1.0e-9 / cost
        #print("a:", a.numpy())
        #print("b:", b.numpy())
        #print("out:", out.numpy())
        print("costs:", costs)
        print("%g secs/op" % cost)
        print("%.3f GFLOPS" % gflops)
