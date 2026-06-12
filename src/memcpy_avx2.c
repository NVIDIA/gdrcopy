/*
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <assert.h>

// src is WC MMIO of GPU BAR
// dest is host memory
int memcpy_uncached_load_avx2(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __AVX2__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((s_int & (sizeof(__m256i)-1)) == 0);
    assert(n >= sizeof(__m256i) && n % sizeof(__m256i) == 0);

    if (d_int & (sizeof(__m256i)-1)) { // dest is not aligned to 256-bits
        __m256i r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256i)) {
            r0 = _mm256_stream_load_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            r1 = _mm256_stream_load_si256 ((__m256i *)(s+1*sizeof(__m256i)));
            r2 = _mm256_stream_load_si256 ((__m256i *)(s+2*sizeof(__m256i)));
            r3 = _mm256_stream_load_si256 ((__m256i *)(s+3*sizeof(__m256i)));
            _mm256_storeu_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            _mm256_storeu_si256((__m256i *)(d+1*sizeof(__m256i)), r1);
            _mm256_storeu_si256((__m256i *)(d+2*sizeof(__m256i)), r2);
            _mm256_storeu_si256((__m256i *)(d+3*sizeof(__m256i)), r3);
            s += 4*sizeof(__m256i);
            d += 4*sizeof(__m256i);
            n -= 4*sizeof(__m256i);
        }
        while (n >= sizeof(__m256i)) {
            r0 = _mm256_stream_load_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            _mm256_storeu_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            s += sizeof(__m256i);
            d += sizeof(__m256i);
            n -= sizeof(__m256i);
        }
    } else { // or it IS aligned
        __m256i r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256i)) {
            r0 = _mm256_stream_load_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            r1 = _mm256_stream_load_si256 ((__m256i *)(s+1*sizeof(__m256i)));
            r2 = _mm256_stream_load_si256 ((__m256i *)(s+2*sizeof(__m256i)));
            r3 = _mm256_stream_load_si256 ((__m256i *)(s+3*sizeof(__m256i)));
            _mm256_store_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            _mm256_store_si256((__m256i *)(d+1*sizeof(__m256i)), r1);
            _mm256_store_si256((__m256i *)(d+2*sizeof(__m256i)), r2);
            _mm256_store_si256((__m256i *)(d+3*sizeof(__m256i)), r3);
            s += 4*sizeof(__m256i);
            d += 4*sizeof(__m256i);
            n -= 4*sizeof(__m256i);
        }
        while (n >= sizeof(__m256i)) {
            r0 = _mm256_stream_load_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            _mm256_store_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            s += sizeof(__m256i);
            d += sizeof(__m256i);
            n -= sizeof(__m256i);
        }
    }

    assert(n == 0);

#else
#error "this file should be compiled with -mavx2"
#endif
    return ret;
}

// dest is WC MMIO of GPU BAR
// src is host memory
int memcpy_uncached_store_avx2(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __AVX2__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((d_int & (sizeof(__m256i)-1)) == 0);
    assert(n >= sizeof(__m256i) && n % sizeof(__m256i) == 0);

    if (s_int & (sizeof(__m256i)-1)) { // source is not aligned to 256-bits
        __m256i r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256i)) {
            r0 = _mm256_loadu_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            r1 = _mm256_loadu_si256 ((__m256i *)(s+1*sizeof(__m256i)));
            r2 = _mm256_loadu_si256 ((__m256i *)(s+2*sizeof(__m256i)));
            r3 = _mm256_loadu_si256 ((__m256i *)(s+3*sizeof(__m256i)));
            _mm256_stream_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            _mm256_stream_si256((__m256i *)(d+1*sizeof(__m256i)), r1);
            _mm256_stream_si256((__m256i *)(d+2*sizeof(__m256i)), r2);
            _mm256_stream_si256((__m256i *)(d+3*sizeof(__m256i)), r3);
            s += 4*sizeof(__m256i);
            d += 4*sizeof(__m256i);
            n -= 4*sizeof(__m256i);
        }
        while (n >= sizeof(__m256i)) {
            r0 = _mm256_loadu_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            _mm256_stream_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            s += sizeof(__m256i);
            d += sizeof(__m256i);
            n -= sizeof(__m256i);
        }
    } else { // or it IS aligned
        __m256i r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256i)) {
            r0 = _mm256_load_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            r1 = _mm256_load_si256 ((__m256i *)(s+1*sizeof(__m256i)));
            r2 = _mm256_load_si256 ((__m256i *)(s+2*sizeof(__m256i)));
            r3 = _mm256_load_si256 ((__m256i *)(s+3*sizeof(__m256i)));
            _mm256_stream_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            _mm256_stream_si256((__m256i *)(d+1*sizeof(__m256i)), r1);
            _mm256_stream_si256((__m256i *)(d+2*sizeof(__m256i)), r2);
            _mm256_stream_si256((__m256i *)(d+3*sizeof(__m256i)), r3);
            s += 4*sizeof(__m256i);
            d += 4*sizeof(__m256i);
            n -= 4*sizeof(__m256i);
        }
        while (n >= sizeof(__m256i)) {
            r0 = _mm256_load_si256 ((__m256i *)(s+0*sizeof(__m256i)));
            _mm256_stream_si256((__m256i *)(d+0*sizeof(__m256i)), r0);
            s += sizeof(__m256i);
            d += sizeof(__m256i);
            n -= sizeof(__m256i);
        }
    }
    // fences are taken care of in the main gdr_copy_to_mapping_internal function

    assert(n == 0);

#else
#error "this file should be compiled with -mavx2"
#endif
    return ret;
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
