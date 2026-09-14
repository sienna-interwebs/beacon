#include <cuda_runtime.h>

extern "C" __global__ void residual_add_fwd(
    char* arena,
    size_t out_off,
    size_t a_off,
    size_t b_off,
    int n) {
  float* out = reinterpret_cast<float*>(arena + out_off);
  const float* a = reinterpret_cast<const float*>(arena + a_off);
  const float* b = reinterpret_cast<const float*>(arena + b_off);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

extern "C" __global__ void residual_add_bwd(
    char* arena,
    size_t da_off,
    size_t db_off,
    size_t dout_off,
    int n) {
  float* da = reinterpret_cast<float*>(arena + da_off);
  float* db = reinterpret_cast<float*>(arena + db_off);
  const float* dout = reinterpret_cast<const float*>(arena + dout_off);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    da[i] += dout[i];
    db[i] += dout[i];
  }
}
