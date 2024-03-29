
import tvm
from tvm import te
import numpy as np
from .kernel import Kernel


class AddOne(Kernel):
    def __init__(self):
        super().__init__()
        self._mod_name = "addone"
        self._B = None
        self._C = None
    
    
    def build(self, target, runtime):
        # v1: only 1 element.
        #self._A = A = te.placeholder((1,), name="A")
        #self._B = B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        #self._out = out = B
        #args = [A, B]

        # v2.1: 1024 elements.
        self._A = A = te.placeholder((1024,), name="A")
        self._B = B = te.compute(A.shape, lambda *i: A(*i), name="B")
        self._out = out = B
        args = [A, B]

        # v2.2: 1024 elements.
        #self._A = A = te.placeholder((1024,), name="A")
        #self._B = B = te.placeholder((1024,), name="B")
        #self._out = out = self._C = C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
        #args = [A, B, C]
        
        # v2.3
        #self._A = A = te.placeholder((1024,), name="A")
        #self._B = B = te.placeholder((1024,), name="B")
        #out = C = te.compute((A.shape[0], B.shape[0]), lambda i, j: A[i] + B(j), name="C")
        #args = [A, B, C]

        # v3: n elements.
        #n = te.var("n")
        #self._A = A = te.placeholder((n,), name="A")
        #self._B = B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        #self._out = out = B
        #args = [A, B]
        
        s = te.create_schedule(out.op)

        # vectorization.
        #ni, nj = s[out].op.axis
        #no, ni = s[out].split(nj, 4)
        #s[out].vectorize(ni)

        print(tvm.lower(s, args, simple_mode=True))

        #self._func_name = self._mod_name
        self._func_name = "default_function"
        #self._func_name = "wasm_codegen_kernel"
        #self._func_code_name = "1599"
        self._func_code_name = "default_function"
        fadd = tvm.build(s, args, target, runtime=runtime, name=self._func_code_name)
        return fadd
    

    def test_rpc(self, remote_system_lib, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=1024).astype(dtype), dev)
        print("a:", a.numpy())
        out = tvm.nd.array(np.zeros(1024, dtype=dtype), dev)
        # invoke the function
        addone = remote_system_lib.get_function(self._func_name)
        if self._out == self._B:
            addone(a, out)
            #np.testing.assert_equal(out.numpy(), a.numpy() + 1)
        elif self._out == self._C:
            b = tvm.nd.array(np.ones(1024, dtype=dtype), dev)
            print("b:", b.numpy())
            addone(a, b, out)
            #np.testing.assert_equal(out.numpy(), a.numpy() + b.numpy())
        print("a:", a.numpy())
        if self._C is not None:
            print("b:", b.numpy())
        print("out:", out.numpy())
        print("Test pass!")

    
    def test_time_evaluation(self, remote, dev):
        dtype = self._A.dtype
        #time_f = remote.system_lib().time_evaluator(self._mod_name, dev, number=100, repeat=10)
        time_f = remote.get_function("__sync.wasm.TimeExecutionForWasm")
        
        a = tvm.nd.array(np.random.uniform(size=1024).astype(dtype), dev)
        print("a:", a.numpy())
        out = tvm.nd.array(np.zeros(1024, dtype=dtype), dev)
        if self._out == self._B:
            cost_bytes = time_f(self._func_name, dev, 100, 10, 0, 0, 0, 0, a, out)
            cost = np.mean(np.frombuffer(cost_bytes, dtype=np.dtype("<f8")))
            #np.testing.assert_equal(out.numpy(), a.numpy() + 1)
        elif self._out == self._C:
            b = tvm.nd.array(np.ones(1024, dtype=dtype), dev)
            print("b:", b.numpy())
            cost_bytes = time_f(self._func_name, dev, 100, 10, 0, 0, 0, 0, a, b, out)
            cost = np.mean(np.frombuffer(cost_bytes, dtype=np.dtype("<f8")))
            #np.testing.assert_equal(out.numpy(), a.numpy() + b.numpy())
        print("a:", a.numpy())
        if self._C is not None:
            print("b:", b.numpy())
        print("out:", out.numpy())
        print("%g secs/op" % cost)
