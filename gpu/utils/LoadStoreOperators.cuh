/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "Float16.cuh"

//
// Templated wrappers to express load/store for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace faiss { namespace gpu {

template <typename T>
struct LoadStore {
  static inline __device__ T load(void* p) {
    return *((T*) p);
  }

  static inline __device__ void store(void* p, const T& v) {
    *((T*) p) = v;
  }
};

#ifdef FAISS_USE_FLOAT16

template <>
struct LoadStore<Half4> {
  static inline __device__ Half4 load(void* p) {
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];" :
        "=h"(HALF_AS_BITS(out.a)), "=h"(HALF_AS_BITS(out.b)) : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];" :
        "=r"(out.a.x), "=r"(out.b.x) : "l"(p));
#endif // CUDA_VERSION
    return out;
  }

  static inline __device__ void store(void* p, const Half4& v) {
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "h"(HALF_AS_BITS(v.a)), "h"(HALF_AS_BITS(v.b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(v.a.x), "r"(v.b.x));
#endif // CUDA_VERSION
  }
};

template <>
struct LoadStore<Half8> {
  static inline __device__ Half8 load(void* p) {
    Half8 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" :
        "=h"(HALF_AS_BITS(out.a.a)), "=h"(HALF_AS_BITS(out.a.b)),
        "=h"(HALF_AS_BITS(out.b.a)), "=h"(HALF_AS_BITS(out.b.b)) : "l"(p));
#else
    asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" :
        "=r"(out.a.a.x), "=r"(out.a.b.x),
        "=r"(out.b.a.x), "=r"(out.b.b.x) : "l"(p));
#endif // CUDA_VERSION
    return out;
  }

  static inline __device__ void store(void* p, const Half8& v) {
#if CUDA_VERSION >= 9000
    asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
        : : "l"(p), "h"(HALF_AS_BITS(v.a.a)), "h"(HALF_AS_BITS(v.a.b)), "h"(HALF_AS_BITS(v.b.a)), "h"(HALF_AS_BITS(v.b.b)));
#else
    asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
        : : "l"(p), "r"(v.a.a.x), "r"(v.a.b.x), "r"(v.b.a.x), "r"(v.b.b.x));
#endif // CUDA_VERSION
  }
};

#endif // FAISS_USE_FLOAT16

} } // namespace
