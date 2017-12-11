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

int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __SSE__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    // align dest to 128-bits
    if (d_int & 0xf) {
        size_t nh = min(0x10 - (d_int & 0x0f), n);
        memcpy(d, s, nh);
        d += nh; d_int += nh;
        s += nh; s_int += nh;
        n -= nh;
    }

    if (s_int & 0xf) { // src is not aligned to 128-bits
        __m128 r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*4*sizeof(float)) {
            r0 = _mm_loadu_ps((float *)(s+0*4*sizeof(float)));
            r1 = _mm_loadu_ps((float *)(s+1*4*sizeof(float)));
            r2 = _mm_loadu_ps((float *)(s+2*4*sizeof(float)));
            r3 = _mm_loadu_ps((float *)(s+3*4*sizeof(float)));
            _mm_stream_ps((float *)(d+0*4*sizeof(float)), r0);
            _mm_stream_ps((float *)(d+1*4*sizeof(float)), r1);
            _mm_stream_ps((float *)(d+2*4*sizeof(float)), r2);
            _mm_stream_ps((float *)(d+3*4*sizeof(float)), r3);
            s += 4*4*sizeof(float);
            d += 4*4*sizeof(float);
            n -= 4*4*sizeof(float);
        }
        while (n >= 4*sizeof(float)) {
            r0 = _mm_loadu_ps((float *)(s));
            _mm_stream_ps((float *)(d), r0);
            s += 4*sizeof(float);
            d += 4*sizeof(float);
            n -= 4*sizeof(float);
        }
    } else { // or it IS aligned
        __m128 r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*4*sizeof(float)) {
            r0 = _mm_load_ps((float *)(s+0*4*sizeof(float)));
            r1 = _mm_load_ps((float *)(s+1*4*sizeof(float)));
            r2 = _mm_load_ps((float *)(s+2*4*sizeof(float)));
            r3 = _mm_load_ps((float *)(s+3*4*sizeof(float)));
            _mm_stream_ps((float *)(d+0*4*sizeof(float)), r0);
            _mm_stream_ps((float *)(d+1*4*sizeof(float)), r1);
            _mm_stream_ps((float *)(d+2*4*sizeof(float)), r2);
            _mm_stream_ps((float *)(d+3*4*sizeof(float)), r3);
            s += 4*4*sizeof(float);
            d += 4*4*sizeof(float);
            n -= 4*4*sizeof(float);
        }
        while (n >= 4*sizeof(float)) {
            r0 = _mm_load_ps((float *)(s));
            _mm_stream_ps((float *)(d), r0);
            s += 4*sizeof(float);
            d += 4*sizeof(float);
            n -= 4*sizeof(float);
        }            
    }

    if (n)
        memcpy(d, s, n);

    // fencing is needed even for plain memcpy(), due to performance
    // being hit by delayed flushing of WC buffers
    _mm_sfence();
#else
#error "this file should be compiled with -msse"
#endif
    return ret;
}

int memcpy_cached_store_sse(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __SSE__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    // align dest to 128-bits
    if (d_int & 0xf) {
        size_t nh = min(0x10 - (d_int & 0x0f), n);
        memcpy(d, s, nh);
        d += nh; d_int += nh;
        s += nh; s_int += nh;
        n -= nh;
    }

    if (s_int & 0xf) { // src is not aligned to 128-bits
        __m128 r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*4*sizeof(float)) {
            r0 = _mm_loadu_ps((float *)(s+0*4*sizeof(float)));
            r1 = _mm_loadu_ps((float *)(s+1*4*sizeof(float)));
            r2 = _mm_loadu_ps((float *)(s+2*4*sizeof(float)));
            r3 = _mm_loadu_ps((float *)(s+3*4*sizeof(float)));
            _mm_store_ps((float *)(d+0*4*sizeof(float)), r0);
            _mm_store_ps((float *)(d+1*4*sizeof(float)), r1);
            _mm_store_ps((float *)(d+2*4*sizeof(float)), r2);
            _mm_store_ps((float *)(d+3*4*sizeof(float)), r3);
            s += 4*4*sizeof(float);
            d += 4*4*sizeof(float);
            n -= 4*4*sizeof(float);
        }
        while (n >= 4*sizeof(float)) {
            r0 = _mm_loadu_ps((float *)(s));
            _mm_store_ps((float *)(d), r0);
            s += 4*sizeof(float);
            d += 4*sizeof(float);
            n -= 4*sizeof(float);
        }
    } else { // or it IS aligned
        __m128 r0,r1,r2,r3;
        // unroll 4
        while (n >= 4*4*sizeof(float)) {
            r0 = _mm_load_ps((float *)(s+0*4*sizeof(float)));
            r1 = _mm_load_ps((float *)(s+1*4*sizeof(float)));
            r2 = _mm_load_ps((float *)(s+2*4*sizeof(float)));
            r3 = _mm_load_ps((float *)(s+3*4*sizeof(float)));
            _mm_store_ps((float *)(d+0*4*sizeof(float)), r0);
            _mm_store_ps((float *)(d+1*4*sizeof(float)), r1);
            _mm_store_ps((float *)(d+2*4*sizeof(float)), r2);
            _mm_store_ps((float *)(d+3*4*sizeof(float)), r3);
            s += 4*4*sizeof(float);
            d += 4*4*sizeof(float);
            n -= 4*4*sizeof(float);
        }
        while (n >= 4*sizeof(float)) {
            r0 = _mm_load_ps((float *)(s));
            _mm_store_ps((float *)(d), r0);
            s += 4*sizeof(float);
            d += 4*sizeof(float);
            n -= 4*sizeof(float);
        }            
    }

    if (n)
        memcpy(d, s, n);

    // fencing because of NT stores
    // potential optimization: issue only when NT stores are actually emitted
    _mm_sfence();

#else
#error "this file should be compiled with -msse"
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
