/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __GDRAPI_INTERNAL_H__
#define __GDRAPI_INTERNAL_H__

#include <stdint.h> // for standard [u]intX_t types
#include <stddef.h>
#include <sys/queue.h>
#include "gdrapi.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef unlikely
#ifdef __GNUC__
#define unlikely(x)         __builtin_expect(!!(x), 0)
#else
#define unlikely(x)         (x)
#endif
#endif

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x)      (*(volatile typeof((x)) *)&(x))
#endif

#ifndef READ_ONCE
#define READ_ONCE(x)        ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v)    (ACCESS_ONCE(x) = (v))
#endif

typedef struct gdr_memh_t { 
    uint32_t handle;
    LIST_ENTRY(gdr_memh_t) entries;
    gdr_mapping_type_t mapping_type;
} gdr_memh_t;

struct gdr {
    int fd;
    LIST_HEAD(memh_list, gdr_memh_t) memhs;
    size_t page_size;
    size_t page_mask;
    uint8_t page_shift;
    uint32_t gdrdrv_version;
};

#ifdef __cplusplus
}
#endif

#endif // __GDRAPI_INTERNAL_H__
