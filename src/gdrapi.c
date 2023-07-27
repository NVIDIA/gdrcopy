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
#include <assert.h>
#include <sys/queue.h>

#include "gdrconfig.h"
#include "gdrapi.h"
#include "gdrdrv.h"
#include "gdrapi_internal.h"

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
            va_end(ap);
        }
    }
}

#define gdr_dbg(FMT, ARGS...)  gdr_msg(GDRCOPY_MSG_DEBUG, "DBG:  " FMT, ## ARGS)
#define gdr_dbgc(C, FMT, ARGS...)  do { static int gdr_dbg_cnt=(C); if (gdr_dbg_cnt) { gdr_dbg(FMT, ## ARGS); --gdr_dbg_cnt; }} while (0)
#define gdr_info(FMT, ARGS...) gdr_msg(GDRCOPY_MSG_INFO,  "INFO: " FMT, ## ARGS)
#define gdr_warn(FMT, ARGS...) gdr_msg(GDRCOPY_MSG_WARN,  "WARN: " FMT, ## ARGS)
#define gdr_err(FMT, ARGS...)  gdr_msg(GDRCOPY_MSG_ERROR, "ERR:  " FMT, ## ARGS)

static gdr_memh_t *to_memh(gdr_mh_t mh) {
    return (gdr_memh_t *)mh.h;
}

static gdr_mh_t from_memh(gdr_memh_t *memh) {
    gdr_mh_t mh;
    mh.h = (unsigned long)memh;
    return mh;
}

static void gdr_init_cpu_flags(void);

static inline int gdr_is_mapped(const gdr_mapping_type_t mapping_type)
{
    return mapping_type != GDR_MAPPING_TYPE_NONE;
}

gdr_t gdr_open(void)
{
    gdr_t g = NULL;
    const char *gdrinode = "/dev/gdrdrv";
    int ret;

    g = calloc(1, sizeof(*g));
    if (!g) {
        gdr_err("error while allocating memory\n");
        return NULL;
    }

    int fd = open(gdrinode, O_RDWR | O_CLOEXEC);
    if (-1 == fd ) {
        ret = errno;
        gdr_err("error opening driver (errno=%d/%s)\n", ret, strerror(ret));
        goto err_mem;
    }

    struct GDRDRV_IOC_GET_VERSION_PARAMS params;
    int retcode = ioctl(fd, GDRDRV_IOC_GET_VERSION, &params);
    if (0 != retcode) {
        ret = errno;
        gdr_err("Error getting the gdrdrv driver version with ioctl error (errno=%d). gdrdrv might be too old.\n", ret);
        goto err_fd;
    }
    if (params.gdrdrv_version < MINIMUM_GDRDRV_VERSION) {
        gdr_err(
            "The minimum required gdrdrv driver version is %d.%d but the current gdrdrv version is %d.%d\n", 
            MINIMUM_GDRDRV_MAJOR_VERSION, 
            MINIMUM_GDRDRV_MINOR_VERSION, 
            params.gdrdrv_version >> MAJOR_VERSION_SHIFT, 
            params.gdrdrv_version & MINOR_VERSION_MASK
        );
        goto err_fd;
    }
    if (params.minimum_gdr_api_version > GDR_API_VERSION) {
        gdr_err(
            "gdrdrv driver requires libgdrapi version %d.%d or above but the current libgdrapi version is %d.%d\n", 
            params.minimum_gdr_api_version >> MAJOR_VERSION_SHIFT, 
            params.minimum_gdr_api_version & MINOR_VERSION_MASK, 
            GDR_API_MAJOR_VERSION, 
            GDR_API_MINOR_VERSION
        );
        goto err_fd;
    }

    g->fd = fd;
    LIST_INIT(&g->memhs);

    gdr_init_cpu_flags();

    // Initialize page_shift, page_size, and page_mask.
    g->page_size = sysconf(_SC_PAGESIZE);
    g->page_mask = ~(g->page_size - 1);

    size_t ps_tmp = g->page_size;
    g->page_shift = -1;
    while (ps_tmp > 0) {
        ++g->page_shift;
        if ((ps_tmp & 0x1) == 1)
            break;
        ps_tmp >>= 1;
    }

    g->gdrdrv_version = params.gdrdrv_version;

    return g;

err_fd:
    close(fd);

err_mem:
    free(g);

    return NULL;
}

int gdr_close(gdr_t g)
{
    int ret = 0;
    int retcode;
    gdr_memh_t *mh, *next_mh;

    mh = g->memhs.lh_first;
    while (mh != NULL) {
        // gdr_unpin_buffer frees mh, so we need to get the next one
        // beforehand.
        next_mh = mh->entries.le_next;
        ret = gdr_unpin_buffer(g, from_memh(mh));
        if (ret) {
            gdr_err("error unpinning buffer inside gdr_close (errno=%d/%s)\n", ret, strerror(ret));
            return ret;
        }
        mh = next_mh;
    }

    retcode = close(g->fd);
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
    LIST_INSERT_HEAD(&g->memhs, mh, entries);
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
    LIST_REMOVE(mh, entries);
    free(mh);
    
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

int gdr_get_info_v2(gdr_t g, gdr_mh_t handle, gdr_info_v2_t *info)
{
    int ret = 0;
    int retcode;
    gdr_memh_t *mh = to_memh(handle);

    if (g->gdrdrv_version >= GDRDRV_MINIMUM_VERSION_WITH_GET_INFO_V2) {
        struct GDRDRV_IOC_GET_INFO_V2_PARAMS params;
        params.handle = mh->handle;

        retcode = ioctl(g->fd, GDRDRV_IOC_GET_INFO_V2, &params);
        if (0 != retcode) {
            ret = errno;
            gdr_err("ioctl error (errno=%d)\n", ret);
            goto out;
        } else {
            info->va            = params.va;
            info->mapped_size   = params.mapped_size;
            info->page_size     = params.page_size;
            info->tm_cycles     = params.tm_cycles;
            info->cycles_per_ms = params.tsc_khz;
            info->mapped        = gdr_is_mapped(params.mapping_type);
            info->wc_mapping    = (params.mapping_type == GDR_MAPPING_TYPE_WC);
            info->mapping_type  = params.mapping_type;
        }
    }
    else
    {
        struct GDRDRV_IOC_GET_INFO_PARAMS params;
        params.handle = mh->handle;

        retcode = ioctl(g->fd, GDRDRV_IOC_GET_INFO, &params);
        if (0 != retcode) {
            ret = errno;
            gdr_err("ioctl error (errno=%d)\n", ret);
            goto out;
        } else {
            info->va            = params.va;
            info->mapped_size   = params.mapped_size;
            info->page_size     = params.page_size;
            info->tm_cycles     = params.tm_cycles;
            info->cycles_per_ms = params.tsc_khz;
            info->mapped        = params.mapped;
            info->wc_mapping    = params.wc_mapping;
            info->mapping_type  = params.mapped ? (params.wc_mapping ? GDR_MAPPING_TYPE_WC : GDR_MAPPING_TYPE_CACHING) : GDR_MAPPING_TYPE_NONE;
        }
    }

out:
    return ret;
}

int gdr_map(gdr_t g, gdr_mh_t handle, void **ptr_va, size_t size)
{
    int ret = 0;
    gdr_info_v2_t info = {0,};
    gdr_memh_t *mh = to_memh(handle);

    if (gdr_is_mapped(mh->mapping_type)) {
        gdr_err("mh is mapped already\n");
        return EAGAIN;
    }
    size_t rounded_size = (size + g->page_size - 1) & g->page_mask;
    off_t magic_off = (off_t)mh->handle << g->page_shift;
    void *mmio = mmap(NULL, rounded_size, PROT_READ|PROT_WRITE, MAP_SHARED, g->fd, magic_off);
    if (mmio == MAP_FAILED) {
        int __errno = errno;
        mmio = NULL;
        gdr_err("error %s(%d) while mapping handle %x, rounded_size=%zu offset=%llx\n",
                strerror(__errno), __errno, handle, rounded_size, (long long unsigned)magic_off);
        ret = __errno;
        goto err;
    }
    *ptr_va = mmio;
    ret = gdr_get_info_v2(g, handle, &info);
    if (ret) {
        gdr_err("error %d from get_info, munmapping before exiting\n", ret);
        munmap(mmio, rounded_size);
        goto err;
    }
    if (!gdr_is_mapped(info.mapping_type)) {
        // Race could cause this issue.
        // E.g., gdr_map and cuMemFree are triggered concurrently.
        // The above mmap is successful but cuMemFree causes unmapping immediately.
        gdr_err("mh is not mapped\n");
        ret = EAGAIN;
    }
    mh->mapping_type = info.mapping_type;
    gdr_dbg("mapping_type=%d\n", mh->mapping_type);
 err:
    return ret;
}

int gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size)
{
    int ret = 0;
    int retcode = 0;
    size_t rounded_size;
    gdr_memh_t *mh = to_memh(handle);

    rounded_size = (size + g->page_size - 1) & g->page_mask;

    if (!gdr_is_mapped(mh->mapping_type)) {
        gdr_err("mh is not mapped yet\n");
        return EINVAL;
    }
    retcode = munmap(va, rounded_size);
    if (-1 == retcode) {
        int __errno = errno;
        gdr_err("error %s(%d) while unmapping handle %x, rounded_size=%zu\n",
                strerror(__errno), __errno, handle, rounded_size);
        ret = __errno;
        goto err;
    }
    mh->mapping_type = GDR_MAPPING_TYPE_NONE;
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
#define PREFERS_STORE_UNROLL4 0
#define PREFERS_STORE_UNROLL8 0
#define PREFERS_LOAD_UNROLL4  0
#define PREFERS_LOAD_UNROLL8  0
// GDRAPI_X86

#elif defined(GDRAPI_POWER)
static int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_cached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_cached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static inline void wc_store_fence(void) { asm volatile("sync") ; }
#define PREFERS_STORE_UNROLL4 1
#define PREFERS_STORE_UNROLL8 0
#define PREFERS_LOAD_UNROLL4  0
#define PREFERS_LOAD_UNROLL8  1
// GDRAPI_POWER

#elif defined(GDRAPI_ARM64)
static int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_cached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_cached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static inline void wc_store_fence(void) { asm volatile("DMB ishld") ; }
#define PREFERS_STORE_UNROLL4 0
#define PREFERS_STORE_UNROLL8 0
#define PREFERS_LOAD_UNROLL4  0
#define PREFERS_LOAD_UNROLL8  0
// GDRAPI_ARM64
#endif

static int has_sse = 0;
static int has_sse2 = 0;
static int has_sse4_1 = 0;
static int has_avx = 0;
static int has_avx2 = 0;

static void gdr_init_cpu_flags(void)
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
}

// note: more than one implementation may be compiled in

static void unroll8_memcpy(void *dst, const void *src, size_t size)
{
    const uint64_t *r = (const uint64_t *)src;
    uint64_t *w = (uint64_t *)dst;
    size_t nw = size / sizeof(*r);
    assert(size % sizeof(*r) == 0);

    while (nw) {
        if (0 == (nw & 3)) {
            uint64_t r0 = r[0];
            uint64_t r1 = r[1];
            uint64_t r2 = r[2];
            uint64_t r3 = r[3];
            w[0] = r0;
            w[1] = r1;
            w[2] = r2;
            w[3] = r3;
            r += 4;
            w += 4;
            nw -= 4;
        } else if (0 == (nw & 1)) {
            uint64_t r0 = r[0];
            uint64_t r1 = r[1];
            w[0] = r0;
            w[1] = r1;
            r += 2;
            w += 2;
            nw -= 2;
        } else {
            w[0] = r[0];
            ++w;
            ++r;
            --nw;
        }
    }
}

static void unroll4_memcpy(void *dst, const void *src, size_t size)
{
    const uint32_t *r = (const uint32_t *)src;
    uint32_t *w = (uint32_t *)dst;
    size_t nw = size / sizeof(*r);
    assert(size % sizeof(*r) == 0);

    while (nw) {
        if (0 == (nw & 3)) {
            uint32_t r0 = r[0];
            uint32_t r1 = r[1];
            uint32_t r2 = r[2];
            uint32_t r3 = r[3];
            w[0] = r0;
            w[1] = r1;
            w[2] = r2;
            w[3] = r3;
            r += 4;
            w += 4;
            nw -= 4;
        } else if (0 == (nw & 1)) {
            uint32_t r0 = r[0];
            uint32_t r1 = r[1];
            w[0] = r0;
            w[1] = r1;
            r += 2;
            w += 2;
            nw -= 2;
        } else {
            w[0] = r[0];
            ++w;
            ++r;
            --nw;
        }
    }
}

static inline int is_aligned(unsigned long value, unsigned powof2)
{
    return ((value & (powof2-1)) == 0);
}

static inline int ptr_is_aligned(const void *ptr, unsigned powof2)
{
    unsigned long addr = (unsigned long)ptr;
    return is_aligned(addr, powof2);
}

static inline void memcpy_to_device_mapping(void *dst, const void *src, size_t size)
{
    size_t remaining_size = size;
    void *curr_map_d_ptr = dst;
    const void *curr_h_ptr = src;
    size_t copy_size = 0;
    while (remaining_size > 0) {
        if (is_aligned(remaining_size, sizeof(uint64_t)) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint64_t)) && ptr_is_aligned(curr_h_ptr, sizeof(uint64_t))) {
            // We have proper alignment. memcpy can be used here. Although
            // unlikely, this might break in the future if the implementation
            // of memcpy changes to generate unaligned access. Still, we choose
            // memcpy because it provides better performance than our simple
            // aligned-access workaround.
            memcpy(curr_map_d_ptr, curr_h_ptr, remaining_size);
            copy_size = remaining_size;
        }
        else if (remaining_size >= sizeof(uint64_t) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint64_t))) {
            // memcpy cannot be used here because its internal
            // implementation may end up in an unaligned access.
            WRITE_ONCE(*(uint64_t *)curr_map_d_ptr, *(uint64_t *)curr_h_ptr);
            copy_size = sizeof(uint64_t);
        }
        else if (remaining_size >= sizeof(uint32_t) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint32_t))) {
            WRITE_ONCE(*(uint32_t *)curr_map_d_ptr, *(uint32_t *)curr_h_ptr);
            copy_size = sizeof(uint32_t);
        }
        else if (remaining_size >= sizeof(uint16_t) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint16_t))) {
            WRITE_ONCE(*(uint16_t *)curr_map_d_ptr, *(uint16_t *)curr_h_ptr);
            copy_size = sizeof(uint16_t);
        }
        else {
            WRITE_ONCE(*(uint8_t *)curr_map_d_ptr, *(uint8_t *)curr_h_ptr);
            copy_size = sizeof(uint8_t);
        }
        remaining_size -= copy_size;
        curr_map_d_ptr = (void *)((uintptr_t)curr_map_d_ptr + copy_size);
        curr_h_ptr = (const void *)((uintptr_t)curr_h_ptr + copy_size);
    }
}

static inline void memcpy_from_device_mapping(void *dst, const void *src, size_t size)
{
    size_t remaining_size = size;
    const void *curr_map_d_ptr = src;
    void *curr_h_ptr = dst;
    size_t copy_size = 0;
    while (remaining_size > 0) {
        if (is_aligned(remaining_size, sizeof(uint64_t)) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint64_t)) && ptr_is_aligned(curr_h_ptr, sizeof(uint64_t))) {
            // We have proper alignment. memcpy can be used here. Although
            // unlikely, this might break in the future if the implementation
            // of memcpy changes to generate unaligned access. Still, we choose
            // memcpy because it provides better performance than our simple
            // aligned-access workaround.
            memcpy(curr_h_ptr, curr_map_d_ptr, remaining_size);
            copy_size = remaining_size;
        }
        else if (remaining_size >= sizeof(uint64_t) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint64_t))) {
            // memcpy cannot be used here because its internal
            // implementation may end up in an unaligned access.
            *(uint64_t *)curr_h_ptr = READ_ONCE(*(uint64_t *)curr_map_d_ptr);
            copy_size = sizeof(uint64_t);
        }
        else if (remaining_size >= sizeof(uint32_t) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint32_t))) {
            *(uint32_t *)curr_h_ptr = READ_ONCE(*(uint32_t *)curr_map_d_ptr);
            copy_size = sizeof(uint32_t);
        }
        else if (remaining_size >= sizeof(uint16_t) && ptr_is_aligned(curr_map_d_ptr, sizeof(uint16_t))) {
            *(uint16_t *)curr_h_ptr = READ_ONCE(*(uint16_t *)curr_map_d_ptr);
            copy_size = sizeof(uint16_t);
        }
        else {
            *(uint8_t *)curr_h_ptr = READ_ONCE(*(uint8_t *)curr_map_d_ptr);
            copy_size = sizeof(uint8_t);
        }
        remaining_size -= copy_size;
        curr_map_d_ptr = (const void *)((uintptr_t)curr_map_d_ptr + copy_size);
        curr_h_ptr = (void *)((uintptr_t)curr_h_ptr + copy_size);
    }
}

static int gdr_copy_to_mapping_internal(void *map_d_ptr, const void *h_ptr, size_t size, gdr_mapping_type_t mapping_type)
{
    const int wc_mapping = (mapping_type == GDR_MAPPING_TYPE_WC);
    const int device_mapping = (mapping_type == GDR_MAPPING_TYPE_DEVICE);
    do {
        // For very small sizes and aligned pointers, we use simple store.
        if (size == sizeof(uint8_t)) {
            WRITE_ONCE(*(uint8_t *)map_d_ptr, *(uint8_t *)h_ptr);
            goto do_fence;
        } else if (size == sizeof(uint16_t) && ptr_is_aligned(map_d_ptr, sizeof(uint16_t))) {
            WRITE_ONCE(*(uint16_t *)map_d_ptr, *(uint16_t *)h_ptr);
            goto do_fence;
        } else if (size == sizeof(uint32_t) && ptr_is_aligned(map_d_ptr, sizeof(uint32_t))) {
            WRITE_ONCE(*(uint32_t *)map_d_ptr, *(uint32_t *)h_ptr);
            goto do_fence;
        } else if (size == sizeof(uint64_t) && ptr_is_aligned(map_d_ptr, sizeof(uint64_t))) {
            WRITE_ONCE(*(uint64_t *)map_d_ptr, *(uint64_t *)h_ptr);
            goto do_fence;
        }

        // pick the most performing implementation compatible with the platform we are running on
        // NOTE: write fences are included in functions below
        if (has_avx) {
            assert(wc_mapping);
            gdr_dbgc(1, "using AVX implementation of gdr_copy_to_mapping\n");
            memcpy_uncached_store_avx(map_d_ptr, h_ptr, size);
            goto out;
        }
        if (has_sse) {
            assert(wc_mapping);
            gdr_dbgc(1, "using SSE implementation of gdr_copy_to_mapping\n");
            memcpy_uncached_store_sse(map_d_ptr, h_ptr, size);
            goto out;
        }

        // on POWER, compiler/libc memcpy is not optimal for MMIO
        // 64bit stores are not better than 32bit ones, so we prefer the latter.
        // NOTE: if preferred but not aligned, a better implementation would still try to
        // use byte sized stores to align map_d_ptr and h_ptr to next word.
        // NOTE2: unroll*_memcpy and memcpy do not include fencing.
        if (wc_mapping && PREFERS_STORE_UNROLL8 && is_aligned(size, 8) && ptr_is_aligned(map_d_ptr, 8) && ptr_is_aligned(h_ptr, 8)) {
            gdr_dbgc(1, "using unroll8_memcpy for gdr_copy_to_mapping\n");
            unroll8_memcpy(map_d_ptr, h_ptr, size);
        } else if (wc_mapping && PREFERS_STORE_UNROLL4 && is_aligned(size, 4) && ptr_is_aligned(map_d_ptr, 4) && ptr_is_aligned(h_ptr, 4)) {
            gdr_dbgc(1, "using unroll4_memcpy for gdr_copy_to_mapping\n");
            unroll4_memcpy(map_d_ptr, h_ptr, size);
        } else if (device_mapping) {
            gdr_dbgc(1, "using device-mapping copy for gdr_copy_to_mapping with device mapping\n");
            memcpy_to_device_mapping(map_d_ptr, h_ptr, size);
        } else {
            gdr_dbgc(1, "fallback to compiler/libc memcpy implementation of gdr_copy_to_mapping\n");
            memcpy(map_d_ptr, h_ptr, size);
        }
    } while (0);

do_fence:
    if (wc_mapping) {
        // fencing is needed even for plain memcpy(), due to performance
        // being hit by delayed flushing of WC buffers
        wc_store_fence();
    }

out:
    return 0;
}

static int gdr_copy_from_mapping_internal(void *h_ptr, const void *map_d_ptr, size_t size, gdr_mapping_type_t mapping_type)
{
    const int wc_mapping = (mapping_type == GDR_MAPPING_TYPE_WC);
    const int device_mapping = (mapping_type == GDR_MAPPING_TYPE_DEVICE);

    do {
        // pick the most performing implementation compatible with the platform we are running on
        if (has_sse4_1) {
            assert(wc_mapping);
            gdr_dbgc(1, "using SSE4_1 implementation of gdr_copy_from_mapping\n");
            memcpy_uncached_load_sse41(h_ptr, map_d_ptr, size);
            break;
        }
        if (has_avx) {
            assert(wc_mapping);
            gdr_dbgc(1, "using AVX implementation of gdr_copy_from_mapping\n");
            memcpy_cached_store_avx(h_ptr, map_d_ptr, size);
            break;
        }
        if (has_sse) {
            assert(wc_mapping);
            gdr_dbgc(1, "using SSE implementation of gdr_copy_from_mapping\n");
            memcpy_cached_store_sse(h_ptr, map_d_ptr, size);
            break;
        }

        // on POWER, compiler memcpy is not optimal for MMIO
        // 64bit loads have 2x the BW of 32bit ones
        if (wc_mapping && PREFERS_LOAD_UNROLL8 && is_aligned(size, 8) && ptr_is_aligned(map_d_ptr, 8) && ptr_is_aligned(h_ptr, 8)) {
            gdr_dbgc(1, "using unroll8_memcpy for gdr_copy_from_mapping\n");
            unroll8_memcpy(h_ptr, map_d_ptr, size);
        } else if (wc_mapping && PREFERS_LOAD_UNROLL4 && is_aligned(size, 4) && ptr_is_aligned(map_d_ptr, 4) && ptr_is_aligned(h_ptr, 4)) {
            gdr_dbgc(1, "using unroll4_memcpy for gdr_copy_from_mapping\n");
            unroll4_memcpy(h_ptr, map_d_ptr, size);
        } else if (device_mapping) {
            gdr_dbgc(1, "using device-mapping copy for gdr_copy_from_mapping\n");
            memcpy_from_device_mapping(h_ptr, map_d_ptr, size);
        } else {
            gdr_dbgc(1, "fallback to compiler/libc memcpy implementation of gdr_copy_from_mapping\n");
            memcpy(h_ptr, map_d_ptr, size);
        }

        // note: fencing is not needed because plain stores are used
        // if non-temporal/uncached stores were used on x86, a proper fence would be needed instead
        // if (wc_mapping)
        //    wc_store_fence();
    } while (0);
    
    return 0;
}

int gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size)
{
    gdr_memh_t *mh = to_memh(handle);
    if (unlikely(!gdr_is_mapped(mh->mapping_type))) {
        gdr_err("mh is not mapped yet\n");
        return EINVAL;
    }
    if (unlikely(size == 0))
        return 0;
    return gdr_copy_to_mapping_internal(map_d_ptr, h_ptr, size, mh->mapping_type);
}

int gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size)
{
    gdr_memh_t *mh = to_memh(handle);
    if (unlikely(!gdr_is_mapped(mh->mapping_type))) {
        gdr_err("mh is not mapped yet\n");
        return EINVAL;
    }
    if (unlikely(size == 0))
        return 0;
    return gdr_copy_from_mapping_internal(h_ptr, map_d_ptr, size, mh->mapping_type);
}


void gdr_runtime_get_version(int *major, int *minor)
{
    *major = GDR_API_MAJOR_VERSION;
    *minor = GDR_API_MINOR_VERSION;
}

int gdr_driver_get_version(gdr_t g, int *major, int *minor)
{
    assert(g != NULL);
    assert(g->fd > 0);

    struct GDRDRV_IOC_GET_VERSION_PARAMS params;
    int retcode = ioctl(g->fd, GDRDRV_IOC_GET_VERSION, &params);
    if (0 != retcode) {
        int ret = errno;
        gdr_err("Error getting the gdrdrv driver version with ioctl error (errno=%d). gdrdrv might be too old.\n", ret);
        return ret;
    }

    *major = params.gdrdrv_version >> MAJOR_VERSION_SHIFT;
    *minor = params.gdrdrv_version & MINOR_VERSION_MASK;

    return 0;
}

// ==============================================================================
// Obsoleted API. Provided for compatibility only.
// ==============================================================================

#ifdef gdr_get_info
#undef gdr_get_info
#endif

typedef struct gdr_info_v1 {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    // tm_cycles and cycles_per_ms are deprecated and will be removed in future.
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
} gdr_info_v1_t;

int gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_v1_t *info)
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
        goto out;
    } else {
        info->va            = params.va;
        info->mapped_size   = params.mapped_size;
        info->page_size     = params.page_size;
        info->tm_cycles     = params.tm_cycles;
        info->cycles_per_ms = params.tsc_khz;
        info->mapped        = params.mapped;
        info->wc_mapping    = params.wc_mapping;
    }

out:
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
