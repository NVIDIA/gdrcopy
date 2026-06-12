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
#include <arm_acle.h>
#include <assert.h>

// dest is WC MMIO of GPU BAR
// src is host memory
int memcpy_uncached_store_st64(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __ARM_FEATURE_LS64
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((d_int & (sizeof(data512_t)-1)) == 0);
    assert(n >= sizeof(data512_t) && n % sizeof(data512_t) == 0);

    // unroll 4
    while (n >= 4*sizeof(data512_t)) {
        data512_t r0 = *(const data512_t *)(s + 0*sizeof(data512_t));
        data512_t r1 = *(const data512_t *)(s + 1*sizeof(data512_t));
        data512_t r2 = *(const data512_t *)(s + 2*sizeof(data512_t));
        data512_t r3 = *(const data512_t *)(s + 3*sizeof(data512_t));
        __arm_st64b(d + 0*sizeof(data512_t), r0);
        __arm_st64b(d + 1*sizeof(data512_t), r1);
        __arm_st64b(d + 2*sizeof(data512_t), r2);
        __arm_st64b(d + 3*sizeof(data512_t), r3);
        s += 4*sizeof(data512_t);
        d += 4*sizeof(data512_t);
        n -= 4*sizeof(data512_t);
    }
    while (n >= sizeof(data512_t)) {
        __arm_st64b(d, *(const data512_t *)s);
        s += sizeof(data512_t);
        d += sizeof(data512_t);
        n -= sizeof(data512_t);
    }
    // fences are taken care of in the main gdr_copy_to_mapping_internal function

    assert(n == 0);

#else
    ret = 1;
#endif
    return ret;
}

// src is WC MMIO of GPU BAR
// dest is host memory
int memcpy_uncached_load_st64(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __ARM_FEATURE_LS64
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((s_int & (sizeof(data512_t)-1)) == 0);
    assert(n >= sizeof(data512_t) && n % sizeof(data512_t) == 0);

    // unroll 4
    while (n >= 4*sizeof(data512_t)) {
        data512_t r0 = __arm_ld64b(s + 0*sizeof(data512_t));
        data512_t r1 = __arm_ld64b(s + 1*sizeof(data512_t));
        data512_t r2 = __arm_ld64b(s + 2*sizeof(data512_t));
        data512_t r3 = __arm_ld64b(s + 3*sizeof(data512_t));
        *(data512_t *)(d + 0*sizeof(data512_t)) = r0;
        *(data512_t *)(d + 1*sizeof(data512_t)) = r1;
        *(data512_t *)(d + 2*sizeof(data512_t)) = r2;
        *(data512_t *)(d + 3*sizeof(data512_t)) = r3;
        s += 4*sizeof(data512_t);
        d += 4*sizeof(data512_t);
        n -= 4*sizeof(data512_t);
    }
    while (n >= sizeof(data512_t)) {
        *(data512_t *)d = __arm_ld64b(s);
        s += sizeof(data512_t);
        d += sizeof(data512_t);
        n -= sizeof(data512_t);
    }
    // fences are taken care of in the main gdr_copy_to_mapping_internal function

    assert(n == 0);

#else
    ret = 1;
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
