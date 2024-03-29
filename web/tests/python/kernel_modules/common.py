
from .addone import AddOne
from .vecfma import VecFMA
from .matmul import MatMul


def get_func_mod(mod_name):
    if mod_name == "addone":
        return AddOne()
    elif mod_name == "vecfma":
        L = 4096 * 4096
        return VecFMA(L)
    elif mod_name == "matmul":
        M = K = N = 1024
        return MatMul(M, K, N, "llvm")
    else:
        raise ValueError("Unknown module name %s" % mod_name)
