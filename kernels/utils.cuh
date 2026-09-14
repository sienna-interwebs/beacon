#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/int.hpp>
#include <cute/tensor.hpp>

namespace beacon {

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  constexpr int warps = (BlockSize + 31) / 32;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) {
    smem[wid] = v;
  }
  __syncthreads();
  v = (threadIdx.x < warps) ? smem[lane] : 0.f;
  if (wid == 0) {
    v = warp_reduce_sum(v);
  }
  return v;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  constexpr int warps = (BlockSize + 31) / 32;
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) {
    smem[wid] = v;
  }
  __syncthreads();
  v = (threadIdx.x < warps) ? smem[lane] : -INFINITY;
  if (wid == 0) {
    v = warp_reduce_max(v);
  }
  return v;
}

__device__ __forceinline__ float fp8e4m3_to_f32(uint8_t b) {
  float sign = (b & 0x80u) ? -1.f : 1.f;
  int exp = (b >> 3) & 0x0f;
  int mant = b & 0x07;
  if (exp == 0x0f && mant == 0x07) {
    return __int_as_float(0x7fc00000u);
  }
  if (exp == 0) {
    return sign * static_cast<float>(mant) * exp2f(-9.f);
  }
  float frac = 1.f + static_cast<float>(mant) / 8.f;
  return sign * frac * exp2f(static_cast<float>(exp - 7));
}

__device__ __forceinline__ uint8_t f32_to_fp8e4m3(float x) {
  if (isnan(x)) {
    return 0xffu;
  }
  uint8_t sign = x < 0.f ? 0x80u : 0x00u;
  float ax = fabsf(x);
  if (ax == 0.f) {
    return sign;
  }
  if (ax >= 448.f) {
    return sign | 0x7eu;
  }
  const float min_normal = exp2f(-6.f);
  if (ax < min_normal) {
    float q = ax * exp2f(9.f);
    int mant = static_cast<int>(roundf(q));
    if (mant > 7) {
      mant = 7;
    }
    return sign | static_cast<uint8_t>(mant);
  }
  int exp = static_cast<int>(floorf(log2f(ax)));
  if (exp < -6) {
    exp = -6;
  }
  float frac = ax * exp2f(-static_cast<float>(exp)) - 1.f;
  int mant = static_cast<int>(roundf(frac * 8.f));
  int biased_exp = exp + 7;
  if (mant == 8) {
    mant = 0;
    biased_exp += 1;
  }
  if (biased_exp >= 0x0f && mant >= 0x07) {
    return sign | 0x7eu;
  }
  return sign | static_cast<uint8_t>((biased_exp << 3) | mant);
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ float half_to_f32(__half x) {
  return __half2float(x);
}

__device__ __forceinline__ __half f32_to_half(float x) {
  return __float2half(x);
}

namespace sm90 {

using GmmaF16F16SS_KK =
    cute::SM90_64x64x16_F16F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
using GmmaF16F16SS_TN =
    cute::SM90_64x64x16_F16F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
using GmmaFp8E4m3F32SS_TN = cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>;

template <class Element, class SmemShape>
__host__ __device__ __forceinline__ auto sw128_smem_layout(SmemShape const& shape) {
  return cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<Element>{}, shape);
}

template <class CopyOp, class GTensor, class SLayout, class TileShape>
__host__ __device__ __forceinline__ auto tma_load_atom(
    CopyOp const& op, GTensor const& gmem, SLayout const& smem, TileShape const& tile) {
  return cute::make_tma_atom(op, gmem, smem, tile);
}

template <class Element, class GTensor, class SLayout, class TileShape>
__host__ __device__ __forceinline__ auto tma_load_atom_sm90(
    GTensor const& gmem, SLayout const& smem, TileShape const& tile) {
  return tma_load_atom(cute::SM90_TMA_LOAD{}, gmem, smem, tile);
}

}

union Vec128 {
  uint4 u;
  float4 f;
};

__device__ __forceinline__ Vec128 load_vec128(const void* ptr) {
  Vec128 v;
  v.u = *reinterpret_cast<const uint4*>(ptr);
  return v;
}

__device__ __forceinline__ void store_vec128(void* ptr, Vec128 v) {
  *reinterpret_cast<uint4*>(ptr) = v.u;
}

template <typename T>
__device__ __forceinline__ void load_vec128_t(const T* ptr, T& a, T& b, T& c, T& d) {
  static_assert(sizeof(T) * 4 == sizeof(Vec128), "load_vec128_t expects 128-bit chunk");
  Vec128 v = load_vec128(ptr);
  const T* lane = reinterpret_cast<const T*>(&v);
  a = lane[0];
  b = lane[1];
  c = lane[2];
  d = lane[3];
}

template <typename T>
__device__ __forceinline__ void store_vec128_t(T* ptr, T a, T b, T c, T d) {
  static_assert(sizeof(T) * 4 == sizeof(Vec128), "store_vec128_t expects 128-bit chunk");
  Vec128 v;
  T* lane = reinterpret_cast<T*>(&v);
  lane[0] = a;
  lane[1] = b;
  lane[2] = c;
  lane[3] = d;
  store_vec128(ptr, v);
}

}
