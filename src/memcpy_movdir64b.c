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

#define MOVDIR64B_SIZE 64

// dest is WC MMIO of GPU BAR
// src is host memory
int memcpy_uncached_store_movdir64b(void *dest, const void *src, size_t n_bytes)
{
    int ret = 0;
#ifdef __MOVDIR64B__
    char *d = (char*)dest;
    uintptr_t d_int = (uintptr_t)d;
    const char *s = (const char *)src;
    // uintptr_t s_int = (uintptr_t)s;
    size_t n = n_bytes;

    assert((d_int & (MOVDIR64B_SIZE-1)) == 0);
    assert(n >= MOVDIR64B_SIZE && n % MOVDIR64B_SIZE == 0);

    // unroll 4
    while (n >= 4*MOVDIR64B_SIZE) {
        _movdir64b(d + 0*MOVDIR64B_SIZE, s + 0*MOVDIR64B_SIZE);
        _movdir64b(d + 1*MOVDIR64B_SIZE, s + 1*MOVDIR64B_SIZE);
        _movdir64b(d + 2*MOVDIR64B_SIZE, s + 2*MOVDIR64B_SIZE);
        _movdir64b(d + 3*MOVDIR64B_SIZE, s + 3*MOVDIR64B_SIZE);
        s += 4*MOVDIR64B_SIZE;
        d += 4*MOVDIR64B_SIZE;
        n -= 4*MOVDIR64B_SIZE;
    }
    while (n >= MOVDIR64B_SIZE) {
        _movdir64b(d, s);
        s += MOVDIR64B_SIZE;
        d += MOVDIR64B_SIZE;
        n -= MOVDIR64B_SIZE;
    }
    // fences are taken care of in the main gdr_copy_to_mapping_internal function

    assert(n == 0);

#else
#error "this file should be compiled with -mmovdir64b"
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
