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
#include <arm_neon.h>
#include <assert.h>

// src is WC MMIO of GPU BAR
// dest is host memory
int memcpy_uncached_load_neon(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
    char *d = (char*)dest;
    // uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((s_int & (sizeof(uint8x16x4_t)-1)) == 0);
    assert(n >= sizeof(uint8x16x4_t) && n % sizeof(uint8x16x4_t) == 0);

    // unroll 4
    while (n >= 4*sizeof(uint8x16x4_t)) {
        uint8x16x4_t r0 = vld1q_u8_x4(s + 0*sizeof(uint8x16x4_t));
        uint8x16x4_t r1 = vld1q_u8_x4(s + 1*sizeof(uint8x16x4_t));
        uint8x16x4_t r2 = vld1q_u8_x4(s + 2*sizeof(uint8x16x4_t));
        uint8x16x4_t r3 = vld1q_u8_x4(s + 3*sizeof(uint8x16x4_t));
        vst1q_u8_x4(d + 0*sizeof(uint8x16x4_t), r0);
        vst1q_u8_x4(d + 1*sizeof(uint8x16x4_t), r1);
        vst1q_u8_x4(d + 2*sizeof(uint8x16x4_t), r2);
        vst1q_u8_x4(d + 3*sizeof(uint8x16x4_t), r3);
        s += 4*sizeof(uint8x16x4_t);
        d += 4*sizeof(uint8x16x4_t);
        n -= 4*sizeof(uint8x16x4_t);
    }
    while (n >= sizeof(uint8x16x4_t)) {
        // NEON doesn't need special compiler flags
        // NEON only requires device memory alignment
        uint8x16x4_t data = vld1q_u8_x4(s);
        vst1q_u8_x4(d, data);
        s += sizeof(uint8x16x4_t);
        d += sizeof(uint8x16x4_t);
        n -= sizeof(uint8x16x4_t);
    }

    assert(n == 0);

    return ret;
}

// dest is WC MMIO of GPU BAR
// src is host memory
int memcpy_uncached_store_neon(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    // uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((d_int & (sizeof(uint8x16x4_t)-1)) == 0);
    assert(n >= sizeof(uint8x16x4_t) && n % sizeof(uint8x16x4_t) == 0);

    // unroll 4
    while (n >= 4*sizeof(uint8x16x4_t)) {
        uint8x16x4_t r0 = vld1q_u8_x4(s + 0*sizeof(uint8x16x4_t));
        uint8x16x4_t r1 = vld1q_u8_x4(s + 1*sizeof(uint8x16x4_t));
        uint8x16x4_t r2 = vld1q_u8_x4(s + 2*sizeof(uint8x16x4_t));
        uint8x16x4_t r3 = vld1q_u8_x4(s + 3*sizeof(uint8x16x4_t));
        vst1q_u8_x4(d + 0*sizeof(uint8x16x4_t), r0);
        vst1q_u8_x4(d + 1*sizeof(uint8x16x4_t), r1);
        vst1q_u8_x4(d + 2*sizeof(uint8x16x4_t), r2);
        vst1q_u8_x4(d + 3*sizeof(uint8x16x4_t), r3);
        s += 4*sizeof(uint8x16x4_t);
        d += 4*sizeof(uint8x16x4_t);
        n -= 4*sizeof(uint8x16x4_t);
    }
    while (n >= sizeof(uint8x16x4_t)) {
        uint8x16x4_t data = vld1q_u8_x4(s);
        vst1q_u8_x4(d, data);
        s += sizeof(uint8x16x4_t);
        d += sizeof(uint8x16x4_t);
        n -= sizeof(uint8x16x4_t);
    }

    assert(n == 0);
    
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
