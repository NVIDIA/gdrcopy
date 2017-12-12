/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

// implementation of copy from BAR using MOVNTDQA 
// suggested by Nicholas Wilt <nwilt@amazon.com>

// src is WC MMIO of GPU BAR
// dest is host memory
int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __SSE4_1__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    // align src to 128-bits
    if (s_int & 0xf) {
        size_t nh = min(0x10 - (s_int & 0x0f), n);
        memcpy(d, s, nh);
        d += nh; d_int += nh;
        s += nh; s_int += nh;
        n -= nh;
    }

    if (d_int & 0xf) { // dest is not aligned to 128-bits
        __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        // unroll 8
        while (n >= 8*sizeof(__m128i)) {
            r0 = _mm_stream_load_si128 ((__m128i *)(s+0*sizeof(__m128i)));
            r1 = _mm_stream_load_si128 ((__m128i *)(s+1*sizeof(__m128i)));
            r2 = _mm_stream_load_si128 ((__m128i *)(s+2*sizeof(__m128i)));
            r3 = _mm_stream_load_si128 ((__m128i *)(s+3*sizeof(__m128i)));
            r4 = _mm_stream_load_si128 ((__m128i *)(s+4*sizeof(__m128i)));
            r5 = _mm_stream_load_si128 ((__m128i *)(s+5*sizeof(__m128i)));
            r6 = _mm_stream_load_si128 ((__m128i *)(s+6*sizeof(__m128i)));
            r7 = _mm_stream_load_si128 ((__m128i *)(s+7*sizeof(__m128i)));
            _mm_storeu_si128((__m128i *)(d+0*sizeof(__m128i)), r0);
            _mm_storeu_si128((__m128i *)(d+1*sizeof(__m128i)), r1);
            _mm_storeu_si128((__m128i *)(d+2*sizeof(__m128i)), r2);
            _mm_storeu_si128((__m128i *)(d+3*sizeof(__m128i)), r3);
            _mm_storeu_si128((__m128i *)(d+4*sizeof(__m128i)), r4);
            _mm_storeu_si128((__m128i *)(d+5*sizeof(__m128i)), r5);
            _mm_storeu_si128((__m128i *)(d+6*sizeof(__m128i)), r6);
            _mm_storeu_si128((__m128i *)(d+7*sizeof(__m128i)), r7);
            s += 8*sizeof(__m128i);
            d += 8*sizeof(__m128i);
            n -= 8*sizeof(__m128i);
        }
        while (n >= sizeof(__m128i)) {
            r0 = _mm_stream_load_si128 ((__m128i *)(s+0*sizeof(__m128i)));
            _mm_storeu_si128((__m128i *)(d+0*sizeof(__m128i)), r0);
            s += sizeof(__m128i);
            d += sizeof(__m128i);
            n -= sizeof(__m128i);
        }
    } else { // or it IS aligned
        __m128i r0,r1,r2,r3,r4,r5,r6,r7;
        // unroll 8
        while (n >= 8*sizeof(__m128i)) {
            r0 = _mm_stream_load_si128 ((__m128i *)(s+0*sizeof(__m128i)));
            r1 = _mm_stream_load_si128 ((__m128i *)(s+1*sizeof(__m128i)));
            r2 = _mm_stream_load_si128 ((__m128i *)(s+2*sizeof(__m128i)));
            r3 = _mm_stream_load_si128 ((__m128i *)(s+3*sizeof(__m128i)));
            r4 = _mm_stream_load_si128 ((__m128i *)(s+4*sizeof(__m128i)));
            r5 = _mm_stream_load_si128 ((__m128i *)(s+5*sizeof(__m128i)));
            r6 = _mm_stream_load_si128 ((__m128i *)(s+6*sizeof(__m128i)));
            r7 = _mm_stream_load_si128 ((__m128i *)(s+7*sizeof(__m128i)));
            _mm_stream_si128((__m128i *)(d+0*sizeof(__m128i)), r0);
            _mm_stream_si128((__m128i *)(d+1*sizeof(__m128i)), r1);
            _mm_stream_si128((__m128i *)(d+2*sizeof(__m128i)), r2);
            _mm_stream_si128((__m128i *)(d+3*sizeof(__m128i)), r3);
            _mm_stream_si128((__m128i *)(d+4*sizeof(__m128i)), r4);
            _mm_stream_si128((__m128i *)(d+5*sizeof(__m128i)), r5);
            _mm_stream_si128((__m128i *)(d+6*sizeof(__m128i)), r6);
            _mm_stream_si128((__m128i *)(d+7*sizeof(__m128i)), r7);
            s += 8*sizeof(__m128i);
            d += 8*sizeof(__m128i);
            n -= 8*sizeof(__m128i);
        }
        while (n >= sizeof(__m128i)) {
            r0 = _mm_stream_load_si128 ((__m128i *)(s+0*sizeof(__m128i)));
            _mm_stream_si128((__m128i *)(d+0*sizeof(__m128i)), r0);
            s += sizeof(__m128i);
            d += sizeof(__m128i);
            n -= sizeof(__m128i);
        }
    }

    if (n)
        memcpy(d, s, n);

    // fencing because of NT stores
    // potential optimization: issue only when NT stores are actually emitted
    _mm_sfence();

#else
#error "this file should be compiled with -msse4.1"
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
