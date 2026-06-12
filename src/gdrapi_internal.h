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

enum gdrcopy_msg_level {
    GDRCOPY_MSG_DEBUG = 1,
    GDRCOPY_MSG_INFO,
    GDRCOPY_MSG_WARN,
    GDRCOPY_MSG_ERROR
};

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
    LIST_ENTRY(gdr_memh_t) entries;
    gdr_mapping_type_t mapping_type;
    union {
        struct {
            uint32_t handle;
        } gdrdrv_memh;
        struct {
            int dma_buf_fd;
            uint64_t va;
            size_t page_offset;
            uint64_t mapped_size;
            uint32_t page_size;
            uint64_t tm_cycles;
            uint32_t cycles_per_ms;
            unsigned mapped:1;
            unsigned wc_mapping:1;
            gdr_mapping_type_t mapping_type;
            void *cpu_mapped_va;
            size_t cpu_mapped_len;
        } dmabuf_memh;
    } backend;
} gdr_memh_t;

struct gdr {
    int fd;
    LIST_HEAD(memh_list, gdr_memh_t) memhs;
    size_t page_size;
    size_t page_mask;
    uint8_t page_shift;
    uint32_t gdrdrv_version;
    enum {
        GDR_USE_GDRDRV = 0,
        GDR_USE_DMABUF = 1,
    } cache_backend;
};

void gdr_msg(enum gdrcopy_msg_level lvl, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#define gdr_dbg(FMT, ARGS...)  gdr_msg(GDRCOPY_MSG_DEBUG, "DBG:  " FMT, ## ARGS)
#define gdr_dbgc(C, FMT, ARGS...)  do { static int gdr_dbg_cnt=(C); if (gdr_dbg_cnt) { gdr_dbg(FMT, ## ARGS); --gdr_dbg_cnt; }} while (0)
#define gdr_info(FMT, ARGS...) gdr_msg(GDRCOPY_MSG_INFO,  "INFO: " FMT, ## ARGS)
#define gdr_warn(FMT, ARGS...) gdr_msg(GDRCOPY_MSG_WARN,  "WARN: " FMT, ## ARGS)
#define gdr_err(FMT, ARGS...)  gdr_msg(GDRCOPY_MSG_ERROR, "ERR:  " FMT, ## ARGS)

#endif // __GDRAPI_INTERNAL_H__
