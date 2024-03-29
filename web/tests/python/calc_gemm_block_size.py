
import math

def calc_gemm_block_size(m, k, n, l1_cache_size, l2_cache_size, dtype_bytes):
  print("m %d, k %d, n %d" % (m, k, n))
  print("l1_cache %d, l2_cache %d" % (l1_cache_size, l2_cache_size))
  
  mr = 4
  nr = 4
  kc = calc_kc = (l1_cache_size // dtype_bytes - (mr * nr)) // (mr + nr)
  if calc_kc > k:
    kc = k
  print("mr %d, kc %d (calc %d), nr %d" % (mr, kc, calc_kc, nr))

  for i in range(1, 4096 + 1):
    #mc = nc = int(math.sqrt((m * n) // i))
    if m % i > 0 or n % i > 0:
      continue
    mc = m // i
    nc = n // i
    required_l2_cache_size = (mc * kc + kc * nc + mc * nc) * dtype_bytes
    if required_l2_cache_size <= l2_cache_size:
      break
  print("required_l2_cache %d" % (required_l2_cache_size))
  print("mc %d, kc %d, nc %d" % (mc, kc, nc))

def calc_gemm_block_size_v2(m, k, n, l1_cache_size, l2_cache_size, dtype_bytes):
  print("m %d, k %d, n %d" % (m, k, n))
  print("l1_cache %d, l2_cache %d" % (l1_cache_size, l2_cache_size))
  
  mr = 4
  nr = 4
  kc = 4
  print("mr %d, kc %d, nr %d" % (mr, kc, nr))

  for i in range(1, 4096 + 1):
    mc = nc = int(math.sqrt((m * n) // i))
    if m % mc > 0 or n % nc > 0:
      continue
    #if m % i > 0 or n % i > 0:
    #  continue
    #mc = m // i
    #nc = n // i
    required_l1_cache_size = (mc * kc + kc * nc + mc * nc) * dtype_bytes
    if required_l1_cache_size <= l1_cache_size:
      break
  print("required_l1_cache %d" % (required_l1_cache_size))
  print("mc %d, kc %d, nc %d" % (mc, kc, nc))

if __name__ == "__main__":
  m = 128 * 128
  k = 64
  n = 256
  l1_cache_size = 64 * 1024
  l2_cache_size = 1024 * 1024
  dtype_bytes = 4
  #calc_gemm_block_size(m, k, n, l1_cache_size, l2_cache_size, dtype_bytes)
  calc_gemm_block_size_v2(m, k, n, l1_cache_size, l2_cache_size, dtype_bytes)
