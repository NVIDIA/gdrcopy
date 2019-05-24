/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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

#ifndef min
#define min(A,B) ((A)<(B)?(A):(B))
#endif

int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __AVX__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    // align dest to 256-bits
    if (d_int & 0x1f) {
        size_t nh = min(0x20 - (d_int & 0x1f), n);
        memcpy(d, s, nh);
        d += nh; d_int += nh;
        s += nh; s_int += nh;
        n -= nh;
    }

    if (s_int & 0x1f) { // src is not aligned to 256-bits
        __m256d r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256d)) {
            r0 = _mm256_loadu_pd((double *)(s+0*sizeof(__m256d)));
            r1 = _mm256_loadu_pd((double *)(s+1*sizeof(__m256d)));
            r2 = _mm256_loadu_pd((double *)(s+2*sizeof(__m256d)));
            r3 = _mm256_loadu_pd((double *)(s+3*sizeof(__m256d)));
            _mm256_stream_pd((double *)(d+0*sizeof(__m256d)), r0);
            _mm256_stream_pd((double *)(d+1*sizeof(__m256d)), r1);
            _mm256_stream_pd((double *)(d+2*sizeof(__m256d)), r2);
            _mm256_stream_pd((double *)(d+3*sizeof(__m256d)), r3);
            s += 4*sizeof(__m256d);
            d += 4*sizeof(__m256d);
            n -= 4*sizeof(__m256d);
        }
        while (n >= sizeof(__m256d)) {
            r0 = _mm256_loadu_pd((double *)(s));
            _mm256_stream_pd((double *)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }
    } else { // or it IS aligned
        __m256d r0,r1,r2,r3,r4,r5,r6,r7;
        // unroll 8
        while (n >= 8*sizeof(__m256d)) {
            r0 = _mm256_load_pd((double *)(s+0*sizeof(__m256d)));
            r1 = _mm256_load_pd((double *)(s+1*sizeof(__m256d)));
            r2 = _mm256_load_pd((double *)(s+2*sizeof(__m256d)));
            r3 = _mm256_load_pd((double *)(s+3*sizeof(__m256d)));
            r4 = _mm256_load_pd((double *)(s+4*sizeof(__m256d)));
            r5 = _mm256_load_pd((double *)(s+5*sizeof(__m256d)));
            r6 = _mm256_load_pd((double *)(s+6*sizeof(__m256d)));
            r7 = _mm256_load_pd((double *)(s+7*sizeof(__m256d)));
            _mm256_stream_pd((double *)(d+0*sizeof(__m256d)), r0);
            _mm256_stream_pd((double *)(d+1*sizeof(__m256d)), r1);
            _mm256_stream_pd((double *)(d+2*sizeof(__m256d)), r2);
            _mm256_stream_pd((double *)(d+3*sizeof(__m256d)), r3);
            _mm256_stream_pd((double *)(d+4*sizeof(__m256d)), r4);
            _mm256_stream_pd((double *)(d+5*sizeof(__m256d)), r5);
            _mm256_stream_pd((double *)(d+6*sizeof(__m256d)), r6);
            _mm256_stream_pd((double *)(d+7*sizeof(__m256d)), r7);
            s += 8*sizeof(__m256d);
            d += 8*sizeof(__m256d);
            n -= 8*sizeof(__m256d);
        }
        while (n >= sizeof(__m256d)) {
            r0 = _mm256_load_pd((double *)(s));
            _mm256_stream_pd((double *)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }            
    }

    if (n)
        memcpy(d, s, n);

    // fencing is needed even for plain memcpy(), due to performance
    // being hit by delayed flushing of WC buffers
    _mm_sfence();

#else
#error "this file should be compiled with -mavx"
#endif
    return ret;
}

int memcpy_cached_store_avx(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __AVX__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    // align dest to 256-bits
    if (d_int & 0x1f) {
        size_t nh = min(0x20 - (d_int & 0x1f), n);
        memcpy(d, s, nh);
        d += nh; d_int += nh;
        s += nh; s_int += nh;
        n -= nh;
    }

    if (s_int & 0x1f) { // src is not aligned to 256-bits
        __m256d r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256d)) {
            r0 = _mm256_loadu_pd((double *)(s+0*sizeof(__m256d)));
            r1 = _mm256_loadu_pd((double *)(s+1*sizeof(__m256d)));
            r2 = _mm256_loadu_pd((double *)(s+2*sizeof(__m256d)));
            r3 = _mm256_loadu_pd((double *)(s+3*sizeof(__m256d)));
            _mm256_store_pd((double *)(d+0*sizeof(__m256d)), r0);
            _mm256_store_pd((double *)(d+1*sizeof(__m256d)), r1);
            _mm256_store_pd((double *)(d+2*sizeof(__m256d)), r2);
            _mm256_store_pd((double *)(d+3*sizeof(__m256d)), r3);
            s += 4*sizeof(__m256d);
            d += 4*sizeof(__m256d);
            n -= 4*sizeof(__m256d);
        }
        while (n >= sizeof(__m256d)) {
            r0 = _mm256_loadu_pd((double *)(s));
            _mm256_store_pd((double *)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }
    } else { // or it IS aligned
        __m256d r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*sizeof(__m256d)) {
            r0 = _mm256_load_pd((double *)(s+0*sizeof(__m256d)));
            r1 = _mm256_load_pd((double *)(s+1*sizeof(__m256d)));
            r2 = _mm256_load_pd((double *)(s+2*sizeof(__m256d)));
            r3 = _mm256_load_pd((double *)(s+3*sizeof(__m256d)));
            _mm256_store_pd((double *)(d+0*sizeof(__m256d)), r0);
            _mm256_store_pd((double *)(d+1*sizeof(__m256d)), r1);
            _mm256_store_pd((double *)(d+2*sizeof(__m256d)), r2);
            _mm256_store_pd((double *)(d+3*sizeof(__m256d)), r3);
            s += 4*sizeof(__m256d);
            d += 4*sizeof(__m256d);
            n -= 4*sizeof(__m256d);
        }
        while (n >= sizeof(__m256d)) {
            r0 = _mm256_load_pd((double *)(s));
            _mm256_store_pd((double *)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }            
    }
    if (n)
        memcpy(d, s, n);

    // fencing is needed because of the use of non-temporal stores
    _mm_sfence();

#else
#error "this file should be compiled with -mavx"
#endif
    return ret;
}

// add variant for _mm_stream_load_si256() / VMOVNTDQA

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
