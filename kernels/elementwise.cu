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

extern "C" __global__ void embedding_lookup_fwd(
    char* arena,
    size_t out_off,
    size_t table_off,
    size_t indices_off,
    int embed_dim,
    int vocab_size,
    int num_tokens) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * embed_dim;
  if (tid >= total) {
    return;
  }
  int token = tid / embed_dim;
  int d = tid % embed_dim;
  const int* indices = reinterpret_cast<const int*>(arena + indices_off);
  int idx = indices[token];
  if (idx < 0 || idx >= vocab_size) {
    return;
  }
  const float* table = reinterpret_cast<const float*>(arena + table_off);
  float* out = reinterpret_cast<float*>(arena + out_off);
  out[tid] = table[static_cast<size_t>(idx) * embed_dim + d];
}
