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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <time.h>
#include <asm/types.h>

#include "gdrapi.h"
#include "gdrdrv.h"
#include "gdrconfig.h"

// based on post at http://stackoverflow.com/questions/3385515/static-assert-in-c
#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(!!(COND))*2-1]
// token pasting madness
#define COMPILE_TIME_ASSERT3(X,L) STATIC_ASSERT(X,static_assertion_at_line_##L)
#define COMPILE_TIME_ASSERT2(X,L) COMPILE_TIME_ASSERT3(X,L)
#define COMPILE_TIME_ASSERT(X)    COMPILE_TIME_ASSERT2(X,__LINE__)

// hint: use page_size = sysconf(_SC_PAGESIZE) instead
#ifdef GDRAPI_POWER
#define PAGE_SHIFT 16
#else // catching all 4KB page size platforms here
#define PAGE_SHIFT 12
#endif
#define PAGE_SIZE  (1UL << PAGE_SHIFT)
#define PAGE_MASK  (~(PAGE_SIZE-1))

// logging/tracing

enum gdrcopy_msg_level {
    GDRCOPY_MSG_DEBUG = 1,
    GDRCOPY_MSG_INFO,
    GDRCOPY_MSG_WARN,
    GDRCOPY_MSG_ERROR
};

static int gdr_msg_level = GDRCOPY_MSG_ERROR;
static int gdr_enable_logging = -1;

static void gdr_msg(enum gdrcopy_msg_level lvl, const char* fmt, ...)
{
    if (-1 == gdr_enable_logging) {
        const char *env = getenv("GDRCOPY_ENABLE_LOGGING");
        if (env)
            gdr_enable_logging = 1;
        else
            gdr_enable_logging = 0;

        env = getenv("GDRCOPY_LOG_LEVEL");
        if (env)
            gdr_msg_level = atoi(env);
    }
    if (gdr_enable_logging) {
        if (lvl >= gdr_msg_level) {
            va_list ap;
            va_start(ap, fmt);
            vfprintf(stderr, fmt, ap);
        }
    }
}

#define gdr_dbg(FMT, ARGS...)  gdr_msg(GDRCOPY_MSG_DEBUG, "DBG:  " FMT, ## ARGS)
#define gdr_dbgc(C, FMT, ARGS...)  do { static int gdr_dbg_cnt=(C); if (gdr_dbg_cnt) { gdr_dbg(FMT, ## ARGS); --gdr_dbg_cnt; }} while (0)
#define gdr_info(FMT, ARGS...) gdr_msg(GDRCOPY_MSG_INFO,  "INFO: " FMT, ## ARGS)
#define gdr_warn(FMT, ARGS...) gdr_msg(GDRCOPY_MSG_WARN,  "WARN: " FMT, ## ARGS)
#define gdr_err(FMT, ARGS...)  gdr_msg(GDRCOPY_MSG_ERROR, "ERR:  " FMT, ## ARGS)

typedef struct { 
    gdr_hnd_t handle;
    unsigned mapped:1;
    unsigned wc_mapping:1;
} gdr_memh_t;

static gdr_memh_t *to_memh(gdr_mh_t mh) {
    return (gdr_memh_t *)mh.h;
}

static gdr_mh_t from_memh(gdr_memh_t *memh) {
    gdr_mh_t mh;
    mh.h = (unsigned long)memh;
    return mh;
}

// check GDR HaNDle size
//COMPILE_TIME_ASSERT(sizeof(gdr_hnd_t)==sizeof(gdr_mh_t));

struct gdr {
    int fd;
};

gdr_t gdr_open()
{
    gdr_t g = NULL;
    const char *gdrinode = "/dev/gdrdrv";

    g = calloc(1, sizeof(*g));
    if (!g) {
        gdr_err("error while allocating memory\n");
        return NULL;
    }

    int fd = open(gdrinode, O_RDWR | O_CLOEXEC);
    if (-1 == fd ) {
        int ret = errno;
        gdr_err("error opening driver (errno=%d/%s)\n", ret, strerror(ret));
        free(g);
        return NULL;
    }

    g->fd = fd;

    return g;
}

int gdr_close(gdr_t g)
{
    int ret = 0;
    int retcode = close(g->fd);
    if (-1 == retcode) {
        ret = errno;
        gdr_err("error closing driver (errno=%d/%s)\n", ret, strerror(ret));
    }
    g->fd = 0;
    free(g);
    return ret;
}

int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle)
{
    int ret = 0;
    int retcode;

    if (!handle) {
        return EINVAL;
    }

    gdr_memh_t *mh = calloc(1, sizeof(gdr_memh_t));
    if (!mh) {
        return ENOMEM;
    }

    struct GDRDRV_IOC_PIN_BUFFER_PARAMS params;
    params.addr = addr;
    params.size = size;
    params.p2p_token = p2p_token;
    params.va_space = va_space;
    params.handle = 0;

    retcode = ioctl(g->fd, GDRDRV_IOC_PIN_BUFFER, &params);
    if (0 != retcode) {
        ret = errno;
        gdr_err("ioctl error (errno=%d)\n", ret);
        goto err;
    }
    mh->handle = params.handle;
    *handle = from_memh(mh);
 err:
    return ret;
}

int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle)
{
    int ret = 0;
    int retcode;
    gdr_memh_t *mh = to_memh(handle);

    struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS params;
    params.handle = mh->handle;
    retcode = ioctl(g->fd, GDRDRV_IOC_UNPIN_BUFFER, &params);
    if (0 != retcode) {
        ret = errno;
        gdr_err("ioctl error (errno=%d)\n", ret);
    }
    
    return ret;
}

int gdr_get_callback_flag(gdr_t g, gdr_mh_t handle, int *flag)
{
    int ret = 0;
    int retcode;
    gdr_memh_t *mh = to_memh(handle);

    struct GDRDRV_IOC_GET_CB_FLAG_PARAMS params;
    params.handle = mh->handle;
    retcode = ioctl(g->fd, GDRDRV_IOC_GET_CB_FLAG, &params);
    if (0 != retcode) {
        ret = errno;
        gdr_err("ioctl error (errno=%d)\n", ret);
    } else {
        *flag = params.flag;
    }
    return ret;
}

int gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info)
{
    int ret = 0;
    int retcode;
    gdr_memh_t *mh = to_memh(handle);

    struct GDRDRV_IOC_GET_INFO_PARAMS params;
    params.handle = mh->handle;

    retcode = ioctl(g->fd, GDRDRV_IOC_GET_INFO, &params);
    if (0 != retcode) {
        ret = errno;
        gdr_err("ioctl error (errno=%d)\n", ret);
    } else {
        info->va          = params.va;
        info->mapped_size = params.mapped_size;
        info->page_size   = params.page_size;
        info->tm_cycles   = params.tm_cycles;
        info->cycles_per_ms = params.tsc_khz;
        info->mapped      = params.mapped;
        info->wc_mapping  = params.wc_mapping;
    }
    return ret;
}

int gdr_map(gdr_t g, gdr_mh_t handle, void **ptr_va, size_t size)
{
    int ret = 0;
    gdr_info_t info = {0,};
    gdr_memh_t *mh = to_memh(handle);

    if (mh->mapped) {
        gdr_err("mh is mapped already\n");
        return EAGAIN;
    }
    size_t rounded_size = (size + PAGE_SIZE - 1) & PAGE_MASK;
    off_t magic_off = (off_t)mh->handle << PAGE_SHIFT;
    void *mmio = mmap(NULL, rounded_size, PROT_READ|PROT_WRITE, MAP_SHARED, g->fd, magic_off);
    if (mmio == MAP_FAILED) {
        int __errno = errno;
        mmio = NULL;
        gdr_err("can't mmap BAR, error=%s(%d) rounded_size=%zu offset=%llx handle=%x\n",
                strerror(__errno), __errno, rounded_size, (long long unsigned)magic_off, handle);
        ret = __errno;
        goto err;
    }
    *ptr_va = mmio;
    ret = gdr_get_info(g, handle, &info);
    if (ret) {
        gdr_err("error %d from get_info, munmapping before exiting\n", ret);
        munmap(mmio, rounded_size);
        goto err;
    }
    mh->mapped = info.mapped;
    if (!mh->mapped) {
        gdr_err("mh should have been mapped at this point\n");
        abort();
    }
    mh->wc_mapping = info.wc_mapping;
    gdr_dbg("wc_mapping=%d\n", mh->wc_mapping);
 err:
    return ret;
}

int gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size)
{
    int ret = 0;
    int retcode = 0;
    size_t rounded_size = (size + PAGE_SIZE - 1) & PAGE_MASK;
    gdr_memh_t *mh = to_memh(handle);

    if (!mh->mapped) {
        gdr_err("mh is not mapped yet\n");
        return EINVAL;
    }
    retcode = munmap(va, rounded_size);
    if (-1 == retcode) {
        int __errno = errno;
        gdr_err("can't unmap BAR, error=%s(%d) rounded_size=%zu\n",
                strerror(__errno), __errno, rounded_size);
        ret = __errno;
        goto err;
    }
    mh->mapped = 0;
    mh->wc_mapping = 0;
 err:
    return ret;
}

#ifdef GDRAPI_X86
#include <cpuid.h>

// prepare for AVX2 implementation
#ifndef bit_AVX2
/* Extended Features (%eax == 7) */
/* %ebx */
#define bit_AVX2 (1 << 5)
#endif

#include <immintrin.h>

extern int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes);
extern int memcpy_cached_store_avx(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes);
extern int memcpy_cached_store_sse(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes);
static inline void wc_store_fence(void) { _mm_sfence(); }
#endif // GDRAPI_X86

#if defined(GDRAPI_POWER)
static int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_cached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_cached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static inline void wc_store_fence(void) { asm volatile("sync") ; }
#endif // GDRAPI_POWER

static int first_time = 1;
static int has_sse = 0;
static int has_sse2 = 0;
static int has_sse4_1 = 0;
static int has_avx = 0;
static int has_avx2 = 0;

static void gdr_init_cpu_flags()
{
#ifdef GDRAPI_X86
    unsigned int info_type = 0x00000001;
    unsigned int ax, bx, cx, dx;
    if (__get_cpuid(info_type, &ax, &bx, &cx, &dx) == 1) {
       has_sse4_1 = ((cx & bit_SSE4_1) != 0);
       has_avx    = ((cx & bit_AVX)    != 0);
       has_sse    = ((dx & bit_SSE)    != 0);
       has_sse2   = ((dx & bit_SSE2)   != 0);
       gdr_dbg("sse4_1=%d avx=%d sse=%d sse2=%d\n", has_sse4_1, has_avx, has_sse, has_sse2);
    }
#ifdef bit_AVX2
    info_type = 0x7;
    if (__get_cpuid(info_type, &ax, &bx, &cx, &dx) == 1) {
        has_avx2 = bx & bit_AVX2;
    }
#endif // bit_AVX2
#endif // GDRAPI_X86

#ifdef GDRAPI_POWER
    // detect and enable Altivec/SMX support
#endif

    first_time = 0;
}

// note: more than one implementation may be compiled in


int gdr_copy_to_bar(void *bar_ptr, const void *h_ptr, size_t size)
{
    if (first_time) {
        gdr_init_cpu_flags();
    }

    do {
        // pick the most performing implementation compatible with the platform we are running on
        if (has_avx) {
            gdr_dbgc(1, "using AVX implementation of gdr_copy_to_bar\n");
            memcpy_uncached_store_avx(bar_ptr, h_ptr, size);
            break;
        }
        if (has_sse) {
            gdr_dbgc(1, "using SSE implementation of gdr_copy_to_bar\n");
            memcpy_uncached_store_sse(bar_ptr, h_ptr, size);
            break;
        }

        // fall through
        gdr_dbgc(1, "using plain implementation of gdr_copy_to_bar\n");
        memcpy(bar_ptr, h_ptr, size);

        // fencing is needed even for plain memcpy(), due to performance
        // being hit by delayed flushing of WC buffers
        wc_store_fence();

    } while (0);

    return 0;
}

int gdr_copy_from_bar(void *h_ptr, const void *bar_ptr, size_t size)
{
    if (first_time) {
        gdr_init_cpu_flags();
    }

    do {
        // pick the most performing implementation compatible with the platform we are running on
        if (has_sse4_1) {
            gdr_dbgc(1, "using SSE4_1 implementation of gdr_copy_from_bar\n");
            memcpy_uncached_load_sse41(h_ptr, bar_ptr, size);
            break;
        }
        if (has_avx) {
            gdr_dbgc(1, "using AVX implementation of gdr_copy_from_bar\n");
            memcpy_cached_store_avx(h_ptr, bar_ptr, size);
            break;
        }
        if (has_sse) {
            gdr_dbgc(1, "using SSE implementation of gdr_copy_from_bar\n");
            memcpy_cached_store_sse(h_ptr, bar_ptr, size);
            break;
        }

        // fall through
        gdr_dbgc(1, "using plain implementation of gdr_copy_from_bar\n");
        memcpy(h_ptr, bar_ptr, size);

        // note: fencing is not needed because plain stores are used
        //wc_store_fence();

    } while (0);

    return 0;
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
