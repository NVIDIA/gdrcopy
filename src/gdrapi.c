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
#include <stdbool.h>
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
#include "cuda_wrapper.h"

// logging/tracing

static int gdr_msg_level = GDRCOPY_MSG_ERROR;
static int gdr_enable_logging = -1;

static int gdr_mapping_type_counters[GDR_MAPPING_TYPE_MAX] = {0,};

// We need a strong fence when mix mapping is active.
static inline bool gdr_has_mix_mapping()
{
    int i;
    int num_type_with_nonzero_counter = 0;
    for (i = 0; i < GDR_MAPPING_TYPE_MAX; ++i) {
        if (gdr_mapping_type_counters[i] != 0)
            ++num_type_with_nonzero_counter;
    }

    return (num_type_with_nonzero_counter > 1);
}

void gdr_msg(enum gdrcopy_msg_level lvl, const char* fmt, ...)
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

static int gdr_open_gdrdrv_internal(gdr_t g)
{
    int fd;
    int retcode, ret;
    const char *gdrinode = "/dev/gdrdrv";

    fd = open(gdrinode, O_RDWR | O_CLOEXEC);
    if (-1 == fd ) {
        ret = errno;
        gdr_err("error opening driver (errno=%d/%s)\n", ret, strerror(ret));
        goto err;
    }

    struct GDRDRV_IOC_GET_VERSION_PARAMS params;
    retcode = ioctl(fd, GDRDRV_IOC_GET_VERSION, &params);
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

    gdr_info("Using gdrdrv backend\n");
    g->gdrdrv_version = params.gdrdrv_version;
    g->cache_backend = GDR_USE_GDRDRV;
    g->fd = fd;

    return 0;

err_fd:
    close(fd);
err:
    return -1;
}

static int gdr_open_dmabuf_internal(gdr_t g)
{
    int supported = 0;
    int status = 0;

    status = gdr_cuda_init();
    if (status){
        gdr_info("gdr_cuda_init() failed\n");
        return status;
    }
    status = gdr_cuda_any_device_supports_dmabuf_mmap(&supported);
    if (status){
        gdr_info("gdr_cuda_any_device_supports_dmabuf_mmap() failed\n");
        goto err_cuda;
    }
    if (!supported) {
        gdr_info("dma-buf mmap not supported on any of the devices\n");
        status = ENOTSUP;
        goto err_cuda;
    }

    gdr_info("Using dma-buf backend\n");
    g->cache_backend = GDR_USE_DMABUF;
    g->fd = -1;

    return 0;

err_cuda:
    gdr_cuda_cleanup();
    return status;
}

gdr_t gdr_open(void)
{
    gdr_t g;
    bool force_dmabuf;
    int ret = 0;

    g = calloc(1, sizeof(*g));
    if (!g) {
        gdr_err("error while allocating memory\n");
        return NULL;
    }

    g->fd = -1;
    const char *env = getenv("GDRCOPY_USE_DMABUF_MMAP");
    force_dmabuf = (env && !(strcmp(env, "1") != 0));

    if (!force_dmabuf) {
        ret = gdr_open_gdrdrv_internal(g);
        if (ret != 0) {
            gdr_info("gdrdrv is not installed or loaded; detecting if CUDA driver supports dmabuf mmap.\n");
        }
    }
    /* If gdrdrv skipped or failed, try dmabuf mmap. */
    if (g->fd == -1) {
        ret = gdr_open_dmabuf_internal(g);
        if (ret != 0) {
            gdr_err("Error: gdrcopy requires gdrdrv or CUDA driver with dmabuf mmap support (at least one is required)\n");
            free(g);
            return NULL;
        }
    }

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

    return g;
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

    if (g->cache_backend == GDR_USE_DMABUF) {
        gdr_cuda_cleanup();
    }
    else {
        retcode = close(g->fd);
        if (-1 == retcode) {
            ret = errno;
            gdr_err("error closing driver (errno=%d/%s)\n", ret, strerror(ret));
            }
        g->fd = 0;
    }
    free(g);
    return ret;
}

static int gdr_pin_buffer_dmabuf(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_memh_t *mh)
{
    int dma_buf_fd = -1, retcode = 0;
    CUdeviceptr aligned_ptr = addr & g->page_mask;
    size_t offset = addr - aligned_ptr;
    size_t aligned_size = (size + offset + g->page_size - 1) & g->page_mask;
    unsigned long long dmabuf_flags = 0llu;
    CUcontext ctx = NULL;
    CUdevice dev;
    int device_ordinal;
    int dmabuf_mmap_supported = 0;
    int retained_primary_ctx = 0;
    int is_coherent = 0;
    gdr_mapping_type_t dmabuf_mapping_type;

    if (flags & GDR_PIN_FLAG_FORCE_PCIE) {
        dmabuf_flags |= CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
    }

    CUCHECK_GOTO(gdr_cuPointerGetAttribute(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, aligned_ptr),
                 retcode, err);
    CUCHECK_GOTO(gdr_cuDeviceGet(&dev, device_ordinal), retcode, err);

    retcode = gdr_cuda_device_supports_dmabuf_mmap(dev, &dmabuf_mmap_supported);
    if (retcode != 0) {
        gdr_err("gdr_cuda_device_supports_dmabuf_mmap failed: %d\n", retcode);
        goto err;
    }
    if (!dmabuf_mmap_supported) {
        gdr_err("device %d does not support dma-buf mmap\n", device_ordinal);
        retcode = ENOTSUP;
        goto err;
    }

    if (gdr_cuda_device_is_coherent(dev, &is_coherent) != 0) {
        is_coherent = 0;
    }
    if (!is_coherent || (flags & GDR_PIN_FLAG_FORCE_PCIE))
        dmabuf_mapping_type = GDR_MAPPING_TYPE_WC;
    else
        dmabuf_mapping_type = GDR_MAPPING_TYPE_CACHING;

    CUCHECK_GOTO(gdr_cuCtxGetCurrent(&ctx), retcode, err);
    if (!ctx) {
        CUCHECK_GOTO(gdr_cuDevicePrimaryCtxRetain(&ctx, dev), retcode, err);
        retained_primary_ctx = 1;
        CUCHECK_GOTO(gdr_cuCtxSetCurrent(ctx), retcode, err);
    }

    CUCHECK_GOTO(gdr_cuMemGetHandleForAddressRange(&dma_buf_fd, aligned_ptr, aligned_size,
                                                   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, dmabuf_flags),
                 retcode, err);

    mh->backend.dmabuf_memh.dma_buf_fd = dma_buf_fd;
    mh->backend.dmabuf_memh.va = aligned_ptr;
    mh->backend.dmabuf_memh.page_offset = offset;
    mh->backend.dmabuf_memh.mapped_size = aligned_size;
    mh->backend.dmabuf_memh.page_size = g->page_size;
    mh->backend.dmabuf_memh.wc_mapping = 0;
    mh->backend.dmabuf_memh.tm_cycles = 0;
    mh->backend.dmabuf_memh.cycles_per_ms = 0;
    mh->backend.dmabuf_memh.mapped = 0;
    mh->backend.dmabuf_memh.mapping_type = dmabuf_mapping_type;
    mh->mapping_type = GDR_MAPPING_TYPE_NONE;

err:
    if (retained_primary_ctx) {
        gdr_cuCtxSetCurrent(NULL);
        gdr_cuDevicePrimaryCtxRelease(dev);
    }
    return retcode;
}

static int gdr_pin_buffer_gdrdrv(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, uint32_t flags, gdr_memh_t *mh)
{
    int retcode = 0;

    if (g->gdrdrv_version >= GDRDRV_MINIMUM_VERSION_WITH_PIN_BUFFER_V2) {
        struct GDRDRV_IOC_PIN_BUFFER_V2_PARAMS params;

        params.addr = addr;
        params.size = size;
        params.flags = 0;
        if (flags & GDR_PIN_FLAG_FORCE_PCIE)
            params.flags |= GDRDRV_PIN_BUFFER_FLAG_FORCE_PCIE;
        params.pad = 0;
        params.handle = 0;

        retcode = ioctl(g->fd, GDRDRV_IOC_PIN_BUFFER_V2, &params);
        if (retcode != 0) {
            gdr_err("ioctl error (errno=%d)\n", errno);
            retcode = errno;
            goto err;
        }
        mh->backend.gdrdrv_memh.handle = params.handle;
    } else {
        struct GDRDRV_IOC_PIN_BUFFER_PARAMS params;

        params.addr = addr;
        params.size = size;
        params.p2p_token = p2p_token;
        params.va_space = va_space;
        params.handle = 0;

        retcode = ioctl(g->fd, GDRDRV_IOC_PIN_BUFFER, &params);
        if (retcode != 0) {
            gdr_err("ioctl error (errno=%d)\n", errno);
            retcode = errno;
            goto err;
        }
        mh->backend.gdrdrv_memh.handle = params.handle;
    }
err:
    return retcode;
}

static int gdr_pin_buffer_internal(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, uint32_t flags, gdr_mh_t *handle){

    int ret = 0;

    if (!handle) {
        return EINVAL;
    }

    gdr_memh_t *mh = calloc(1, sizeof(gdr_memh_t));
    if (!mh) {
        return ENOMEM;
    }

    if (g->cache_backend == GDR_USE_DMABUF){
        ret = gdr_pin_buffer_dmabuf(g, addr, size, flags, mh);
    } else if (g->gdrdrv_version >= GDRDRV_MINIMUM_VERSION_WITH_PIN_BUFFER_V2 ||
               flags == GDR_PIN_FLAG_DEFAULT) {
        ret = gdr_pin_buffer_gdrdrv(g, addr, size, p2p_token, va_space, flags, mh);
    } else {
        gdr_err("gdrdrv is too old and does not support the requested feature\n");
        ret = EINVAL;
    }
    if (ret != 0) {
        free(mh);
        goto err;
    }

    LIST_INSERT_HEAD(&g->memhs, mh, entries);
    *handle = from_memh(mh);
 err:
    return ret;
}

int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle)
{
    return gdr_pin_buffer_internal(g, addr, size, p2p_token, va_space, GDR_PIN_FLAG_DEFAULT, handle);
}

int gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t *handle)
{
    return gdr_pin_buffer_internal(g, addr, size, 0, 0, flags, handle);
}

int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle)
{
    int ret = 0;
    int retcode;
    gdr_memh_t *mh = to_memh(handle);

    if (g->cache_backend == GDR_USE_DMABUF) {
        if (gdr_is_mapped(mh->mapping_type)) {
            gdr_unmap(g, handle, mh->backend.dmabuf_memh.cpu_mapped_va, mh->backend.dmabuf_memh.cpu_mapped_len);
        }
        close(mh->backend.dmabuf_memh.dma_buf_fd);
    }
    else {
        struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS params;
        params.handle = mh->backend.gdrdrv_memh.handle;
        retcode = ioctl(g->fd, GDRDRV_IOC_UNPIN_BUFFER, &params);
        if (0 != retcode) {
            ret = errno;
            gdr_err("ioctl error (errno=%d)\n", ret);
        }
    }
    LIST_REMOVE(mh, entries);
    free(mh);
    
    return ret;
}

int gdr_get_callback_flag(gdr_t g, gdr_mh_t handle, int *flag)
{
    int ret = 0;
    int retcode;

    if (g->cache_backend == GDR_USE_DMABUF) {
        *flag = 0;
    }
    else {
        gdr_memh_t *mh = to_memh(handle);
        struct GDRDRV_IOC_GET_CB_FLAG_PARAMS params;
        params.handle = mh->backend.gdrdrv_memh.handle;
        retcode = ioctl(g->fd, GDRDRV_IOC_GET_CB_FLAG, &params);
        if (0 != retcode) {
            ret = errno;
            gdr_err("ioctl error (errno=%d)\n", ret);
        } else {
            *flag = params.flag;
        }
    }
    return ret;
}

static int gdr_get_info_dmabuf(gdr_t g, gdr_mh_t handle, gdr_info_v2_t *info)
{
    gdr_memh_t *mh = to_memh(handle);

    info->va = mh->backend.dmabuf_memh.va;
    info->mapped_size = mh->backend.dmabuf_memh.mapped_size;
    info->page_size = mh->backend.dmabuf_memh.page_size;
    info->mapping_type = mh->mapping_type;
    info->mapped = gdr_is_mapped(mh->mapping_type);
    info->wc_mapping = mh->backend.dmabuf_memh.wc_mapping;
    info->tm_cycles = mh->backend.dmabuf_memh.tm_cycles;
    info->cycles_per_ms = mh->backend.dmabuf_memh.cycles_per_ms;
    return 0;
}

static int gdr_get_info_gdrdrv(gdr_t g, gdr_mh_t handle, gdr_info_v2_t *info)
{
    int retcode, ret = 0;
    gdr_memh_t *mh = to_memh(handle);

    if (g->gdrdrv_version >= GDRDRV_MINIMUM_VERSION_WITH_GET_INFO_V2) {

        struct GDRDRV_IOC_GET_INFO_V2_PARAMS params;
        params.handle = mh->backend.gdrdrv_memh.handle;

        retcode = ioctl(g->fd, GDRDRV_IOC_GET_INFO_V2, &params);
        if (retcode != 0) {
            ret = errno;
            gdr_err("ioctl error (errno=%d)\n", ret);
            goto err;
        }
        info->va = params.va;
        info->mapped_size = params.mapped_size;
        info->page_size = params.page_size;
        info->tm_cycles = params.tm_cycles;
        info->cycles_per_ms = params.tsc_khz;
        info->mapped = gdr_is_mapped(params.mapping_type);
        info->wc_mapping = (params.mapping_type == GDR_MAPPING_TYPE_WC);
        info->mapping_type = params.mapping_type;
    }
    else {

        struct GDRDRV_IOC_GET_INFO_PARAMS params;
        params.handle = mh->backend.gdrdrv_memh.handle;

        retcode = ioctl(g->fd, GDRDRV_IOC_GET_INFO, &params);
        if (retcode != 0) {
            ret = errno;
            gdr_err("ioctl error (errno=%d)\n", ret);
            goto err;
        }
        info->va = params.va;
        info->mapped_size = params.mapped_size;
        info->page_size = params.page_size;
        info->tm_cycles = params.tm_cycles;
        info->cycles_per_ms = params.tsc_khz;
        info->mapped = params.mapped;
        info->wc_mapping = params.wc_mapping;
        info->mapping_type = params.mapped ? (params.wc_mapping ? GDR_MAPPING_TYPE_WC : GDR_MAPPING_TYPE_CACHING) : GDR_MAPPING_TYPE_NONE;
    }
err:
    return ret;
}

int gdr_get_info_v2(gdr_t g, gdr_mh_t handle, gdr_info_v2_t *info)
{
    int ret = 0;

    if (g->cache_backend == GDR_USE_DMABUF) {
        ret = gdr_get_info_dmabuf(g, handle, info);
    } else {
        ret = gdr_get_info_gdrdrv(g, handle, info);
    }

    return ret;
}

static int gdr_map_gdrdrv(gdr_t g, gdr_memh_t *mh, void **ptr_va,
                             size_t size, int flags,
                             const struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS *params)
{
    int status = 0;
    int retcode;
    gdr_info_v2_t info = {0,};
    gdr_mh_t handle = from_memh(mh);
    size_t rounded_size;
    off_t magic_off;
    void *mmio = NULL;

    if (g->gdrdrv_version >= GDRDRV_MINIMUM_VERSION_WITH_REQ_MAPPING_TYPE) {
        struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS req_params = *params;
        req_params.handle = mh->backend.gdrdrv_memh.handle;
        retcode = ioctl(g->fd, GDRDRV_IOC_REQ_MAPPING_TYPE, &req_params);
        if (0 != retcode) {
            status = errno;
            gdr_err("ioctl error (errno=%d)\n", status);
            goto out;
        }
    }

    rounded_size = (size + g->page_size - 1) & g->page_mask;
    magic_off = (off_t)mh->backend.gdrdrv_memh.handle << g->page_shift;
    mmio = mmap(NULL, rounded_size, PROT_READ | PROT_WRITE, MAP_SHARED, g->fd, magic_off);
    if (mmio == MAP_FAILED) {
        status = errno;
        gdr_err("error %s(%d) while mapping handle %x, rounded_size=%zu offset=%llx\n",
                strerror(status), status, handle, rounded_size, (long long unsigned)magic_off);
        goto out;
    }
    *ptr_va = mmio;
    gdr_dbg("ptr_va: %p\n", *ptr_va);

    status = gdr_get_info_v2(g, handle, &info);
    if (status) {
        gdr_err("error %d from get_info, munmapping before exiting\n", status);
        goto out;
    }

    if (!gdr_is_mapped(info.mapping_type)) {
        gdr_err("mh is not mapped\n");
        status = EAGAIN;
        goto out;
    }

    if (flags != GDR_MAP_FLAG_DEFAULT && info.mapping_type != params->mapping_type) {
        gdr_err("gdrdrv cannot fulfill the requested mapping type. It might be too old.\n");
        status = EINVAL;
        goto out;
    }

    mh->mapping_type = info.mapping_type;
    gdr_dbg("mapping_type=%d\n", mh->mapping_type);

    // GDRCopy is not thread-safe. We use normal increment instead of atomic.
    ++gdr_mapping_type_counters[mh->mapping_type];

out:
    if (status) {
        if (mmio)
            munmap(mmio, rounded_size);
    }
    return status;
}

static int gdr_map_dmabuf(gdr_t g, gdr_memh_t *mh, void **ptr_va, size_t size, int flags,
                             const struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS *params)
{
    int status = 0;
    void *mmio = NULL;
    off_t magic_off = mh->backend.dmabuf_memh.page_offset;
    size_t rounded_size = (size + g->page_size - 1) & g->page_mask;

    mmio = mmap(NULL, rounded_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                mh->backend.dmabuf_memh.dma_buf_fd, magic_off);
    if (mmio == MAP_FAILED) {
        status = errno;
        gdr_err("error %s(%d) while mapping dma_buf_fd, rounded_size=%zu\n",
                strerror(status), status, rounded_size);
        goto out;
    }
    *ptr_va = mmio;
    gdr_dbg("ptr_va: %p\n", *ptr_va);

    gdr_mapping_type_t dmabuf_mapping_type = mh->backend.dmabuf_memh.mapping_type;

    if (params->mapping_type != GDR_MR_NONE && (gdr_mapping_type_t)params->mapping_type != dmabuf_mapping_type) {
        gdr_err("dmabuf backend cannot fulfill the requested mapping type "
                "(requested=%d, dmabuf mmap supports %d).\n",
                (int)params->mapping_type, (int)dmabuf_mapping_type);
        status = EINVAL;
        goto out;
    }
    mh->mapping_type = dmabuf_mapping_type;
    mh->backend.dmabuf_memh.wc_mapping = (dmabuf_mapping_type == GDR_MAPPING_TYPE_WC);
    mh->backend.dmabuf_memh.mapped = 1;
    mh->backend.dmabuf_memh.cpu_mapped_va = mmio;
    mh->backend.dmabuf_memh.cpu_mapped_len = rounded_size;

    // GDRCopy is not thread-safe. We use normal increment instead of atomic.
    ++gdr_mapping_type_counters[mh->mapping_type];

out:
    if (status) {
        if (mmio)
            munmap(mmio, rounded_size);
    }
    return status;
}

int gdr_map_v2(gdr_t g, gdr_mh_t handle, void **ptr_va, size_t size, int flags)
{
    int status = 0;
    gdr_memh_t *mh = to_memh(handle);
    struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS params;

    if (gdr_is_mapped(mh->mapping_type)) {
        gdr_err("mh is mapped already\n");
        return EAGAIN;
    }

    switch (flags) {
        case GDR_MAP_FLAG_DEFAULT:
            params.mapping_type = GDR_MR_NONE;
            break;
        case GDR_MAP_FLAG_REQ_WC_MAPPING:
            params.mapping_type = GDR_MR_WC;
            break;
        case GDR_MAP_FLAG_REQ_CACHE_MAPPING:
            params.mapping_type = GDR_MR_CACHING;
            break;
        case GDR_MAP_FLAG_REQ_DEVICE_MAPPING:
            params.mapping_type = GDR_MR_DEVICE;
            break;
        default:
            gdr_err("encounter unsupported gdr_map_flags\n");
            return EINVAL;
    }

    if (g->cache_backend == GDR_USE_GDRDRV) {
        status = gdr_map_gdrdrv(g, mh, ptr_va, size, flags, &params);
    } else {
        status = gdr_map_dmabuf(g, mh, ptr_va, size, flags, &params);
    }
    return status;
}

int gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size)
{
    return gdr_map_v2(g, handle, va, size, GDR_MAP_FLAG_DEFAULT);
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
    gdr_dbg("unmapping va: %p\n", va);
    retcode = munmap(va, rounded_size);

    if (-1 == retcode) {
        int __errno = errno;
        gdr_err("error %s(%d) while unmapping handle %x, rounded_size=%zu\n",
                strerror(__errno), __errno, handle, rounded_size);
        ret = __errno;
        goto err;
    }

    // gdr_mapping_type_counters may not reflect the number of mappings on this
    // system. For example, users may rely on gdrdrv to automatically unmap
    // when calling cudaFree (non-persistent mapping) or gdr_unpin_buffer. We
    // treat those cases as a user error.
    --gdr_mapping_type_counters[mh->mapping_type];

    mh->mapping_type = GDR_MAPPING_TYPE_NONE;

    if (g->cache_backend == GDR_USE_DMABUF) {
        mh->backend.dmabuf_memh.mapped = 0;
        mh->backend.dmabuf_memh.wc_mapping = 0;
        mh->backend.dmabuf_memh.cpu_mapped_va = NULL;
        mh->backend.dmabuf_memh.cpu_mapped_len = 0;
    }
 err:
    return ret;
}

typedef int (*gdr_copy_fn_t)(void *dest, const void *src, size_t n_bytes);
static gdr_copy_fn_t memcpy_uncached_store_16B = NULL;
static gdr_copy_fn_t memcpy_uncached_store_32B = NULL;
static gdr_copy_fn_t memcpy_uncached_store_64B = NULL;
static gdr_copy_fn_t memcpy_uncached_load_16B = NULL;
static gdr_copy_fn_t memcpy_uncached_load_32B = NULL;
static gdr_copy_fn_t memcpy_uncached_load_64B = NULL;

static const char *memcpy_uncached_store_16B_name = NULL;
static const char *memcpy_uncached_store_32B_name = NULL;
static const char *memcpy_uncached_store_64B_name = NULL;
static const char *memcpy_uncached_load_16B_name = NULL;
static const char *memcpy_uncached_load_32B_name = NULL;
static const char *memcpy_uncached_load_64B_name = NULL;

#ifdef GDRAPI_X86
#include <cpuid.h>

#ifndef bit_SSE4_1
#define bit_SSE4_1 (1 << 19)
#endif
#ifndef bit_OSXSAVE
#define bit_OSXSAVE (1 << 27)
#endif
#ifndef bit_AVX2
#define bit_AVX2 (1 << 5)
#endif
#ifndef bit_AVX512F
#define bit_AVX512F (1 << 16)
#endif
#ifndef bit_MOVDIRI
#define bit_MOVDIRI (1 << 27)
#endif
#ifndef bit_MOVDIR64B
#define bit_MOVDIR64B (1 << 28)
#endif
#ifndef COMPILED_MOVDIR64B
#define COMPILED_MOVDIR64B 0
#endif
#include <immintrin.h>

extern int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_avx(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_sse(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_store_sse41(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_store_avx2(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_avx2(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_store_avx512(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_avx512(void *dest, const void *src, size_t n_bytes);
#if COMPILED_MOVDIR64B
extern int memcpy_uncached_store_movdir64b(void *dest, const void *src, size_t n_bytes);
#else
static int memcpy_uncached_store_movdir64b(void *dest, const void *src, size_t n_bytes) { return 1; }
#endif
static int memcpy_uncached_store_neon(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_neon(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_ls64(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_ls64(void *dest, const void *src, size_t n_bytes) { return 1; }
static inline void wc_store_fence(void) { _mm_sfence(); }
static inline void memory_fence(void) { _mm_mfence() ; }
#define PREFERS_STORE_UNROLL4 0
#define PREFERS_STORE_UNROLL8 0
#define PREFERS_LOAD_UNROLL4  0
#define PREFERS_LOAD_UNROLL8  0
// GDRAPI_X86

#elif defined(GDRAPI_POWER)
static int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_load_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_load_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_store_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_avx2(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_avx2(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_avx512(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_avx512(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_movdir64b(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_neon(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_neon(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_ls64(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_ls64(void *dest, const void *src, size_t n_bytes) { return 1; }
static inline void wc_store_fence(void) { asm volatile("sync") ; }
static inline void memory_fence(void) { asm volatile("sync") ; }
#define PREFERS_STORE_UNROLL4 1
#define PREFERS_STORE_UNROLL8 0
#define PREFERS_LOAD_UNROLL4  0
#define PREFERS_LOAD_UNROLL8  1
// GDRAPI_POWER

#elif defined(GDRAPI_ARM64)
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1 << 1)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef AT_HWCAP3
#define AT_HWCAP3 29
#endif
#ifndef HWCAP3_LS64
#define HWCAP3_LS64 (1UL << 3)
#endif
#ifndef COMPILED_LS64
#define COMPILED_LS64 0
#endif

#include <sys/auxv.h>
#include <asm/hwcap.h>
static int memcpy_uncached_store_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_load_avx(void *dest, const void *src, size_t n_bytes)  { return 1; }
static int memcpy_uncached_store_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_load_sse(void *dest, const void *src, size_t n_bytes)    { return 1; }
static int memcpy_uncached_store_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_sse41(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_avx2(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_avx2(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_avx512(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_avx512(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_store_movdir64b(void *dest, const void *src, size_t n_bytes) { return 1; }
extern int memcpy_uncached_store_neon(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_neon(void *dest, const void *src, size_t n_bytes);
#if COMPILED_LS64
extern int memcpy_uncached_store_ls64(void *dest, const void *src, size_t n_bytes);
extern int memcpy_uncached_load_ls64(void *dest, const void *src, size_t n_bytes);
#else
static int memcpy_uncached_store_ls64(void *dest, const void *src, size_t n_bytes) { return 1; }
static int memcpy_uncached_load_ls64(void *dest, const void *src, size_t n_bytes) { return 1; }
#endif
static inline void wc_store_fence(void) { asm volatile("DMB st") ; }
static inline void memory_fence(void) { asm volatile("DMB sy") ; }
typedef unsigned __int128 uint128_t;
#define PREFERS_STORE_UNROLL4 0
#define PREFERS_STORE_UNROLL8 0
#define PREFERS_LOAD_UNROLL4  0
#define PREFERS_LOAD_UNROLL8  0

static int memcpy_uncached_store_arm64(void *dest, const void *src, size_t n_bytes)
{
    while (n_bytes > 0) {
        WRITE_ONCE(*(uint128_t *)dest, *(const uint128_t *)src);
        dest += 16;
        src += 16;
        n_bytes -= 16;
    }
    assert(n_bytes == 0);
    return 0;
}

static int memcpy_uncached_load_arm64(void *dest, const void *src, size_t n_bytes)
{
    while (n_bytes > 0) {
        *(uint128_t *)dest = READ_ONCE(*(const uint128_t *)src);
        dest += 16;
        src += 16;
        n_bytes -= 16;
    }
    assert(n_bytes == 0);
    return 0;
}

// GDRAPI_ARM64
#endif

static int has_sse = 0;
static int has_sse2 = 0;
static int has_sse4_1 = 0;
static int has_avx = 0;
static int has_osxsave = 0;
static int has_avx2 = 0;
static int has_avx512 = 0;
static int has_movdiri = 0;
static int has_movdir64b = 0;
static int use_intel_avx512 = 0;

static int has_sve = 0;
static int has_neon = 0;
static int has_ls64 = 0;

static void gdr_init_cpu_flags(void)
{
#ifdef GDRAPI_X86
    unsigned int info_type = 0x00000001;
    unsigned int ax, bx, cx, dx;
    uint64_t xcr0 = 0;
    if (__get_cpuid(info_type, &ax, &bx, &cx, &dx) == 1) {
       has_sse4_1 = ((cx & bit_SSE4_1) != 0);
       has_avx    = ((cx & bit_AVX)    != 0);
       has_sse    = ((dx & bit_SSE)    != 0);
       has_sse2   = ((dx & bit_SSE2)   != 0);
       has_osxsave= ((cx & bit_OSXSAVE)!= 0);
       if(has_osxsave){
            uint32_t eax, edx;
            __asm__ __volatile__(
                "xgetbv" 
                : "=a"(eax), "=d"(edx) 
                : "c"(0)
            );
            xcr0 = ((uint64_t)edx << 32) | eax;
       }
       has_avx = has_avx && ((xcr0 & 0x6) != 0);
    }
    info_type = 0x7;
    if (__get_cpuid_count(info_type, 0, &ax, &bx, &cx, &dx) == 1){
        has_avx2 = ((bx & bit_AVX2) != 0) & ((xcr0 & 0x6) != 0);
        has_avx512 = ((bx & bit_AVX512F) != 0) & ((xcr0 & 0xE6) != 0);
        has_movdiri = ((cx & bit_MOVDIRI) != 0);
        has_movdir64b = ((cx & bit_MOVDIR64B) != 0);
        if(has_movdir64b && COMPILED_MOVDIR64B == 0){
            gdr_warn("CPU supports MOVDIR64B but the current binary is not compiled with MOVDIR64B support. Disabling MOVDIR64B usage.\n");
            has_movdir64b = 0;
        }
    }
    info_type = 0x0;
    if (__get_cpuid(info_type, &ax, &bx, &cx, &dx) == 1) {
        char vendor[13];
        *((unsigned int *)&vendor[0]) = bx;
        *((unsigned int *)&vendor[4]) = dx;
        *((unsigned int *)&vendor[8]) = cx;
        vendor[12] = '\0';
        use_intel_avx512 = (strcmp(vendor, "GenuineIntel") == 0);
    }
    gdr_dbg("vendor_intel=%d movdir64b=%d movdiri=%d avx512=%d avx2=%d sse4_1=%d avx=%d sse=%d sse2=%d\n", use_intel_avx512, has_movdir64b, has_movdiri, has_avx512, has_avx2, has_sse4_1, has_avx, has_sse, has_sse2);

    if (has_sse4_1) {
        memcpy_uncached_store_16B = memcpy_uncached_store_sse41;
        memcpy_uncached_store_16B_name = "SSE41";
        memcpy_uncached_load_16B = memcpy_uncached_load_sse41;
        memcpy_uncached_load_16B_name = "SSE41";
    } else if (has_sse) {
        memcpy_uncached_store_16B = memcpy_uncached_store_sse;
        memcpy_uncached_store_16B_name = "SSE";
        memcpy_uncached_load_16B = memcpy_uncached_load_sse;
        memcpy_uncached_load_16B_name = "SSE";
    }

    if (has_avx2) {
        memcpy_uncached_store_32B = memcpy_uncached_store_avx2;
        memcpy_uncached_store_32B_name = "AVX2";
        memcpy_uncached_load_32B = memcpy_uncached_load_avx2;
        memcpy_uncached_load_32B_name = "AVX2";
    } else if (has_avx) {
        memcpy_uncached_store_32B = memcpy_uncached_store_avx;
        memcpy_uncached_store_32B_name = "AVX";
        memcpy_uncached_load_32B = memcpy_uncached_load_avx;
        memcpy_uncached_load_32B_name = "AVX";
    }

    // AMD AVX-512 implementation is not as performant, so we avoid it
    // On many Intel CPUs, SSE4.1 outperforms AVX512, disabling for now 
    // if (has_avx512 && use_intel_avx512) {
    //     memcpy_uncached_store_64B = memcpy_uncached_store_avx512;
    //     memcpy_uncached_store_64B_name = "AVX512";
    //     memcpy_uncached_load_64B = memcpy_uncached_load_avx512;
    //     memcpy_uncached_load_64B_name = "AVX512";
    // }
    if (has_movdir64b) {
        memcpy_uncached_store_64B = memcpy_uncached_store_movdir64b;
        memcpy_uncached_store_64B_name = "MOVDIR64B";
    }
#endif // GDRAPI_X86

#ifdef GDRAPI_ARM64
    unsigned long hwcap = getauxval(AT_HWCAP);
    has_neon = ((hwcap & HWCAP_ASIMD) != 0);
    has_sve = ((hwcap & HWCAP_SVE) != 0);
    unsigned long hwcap3 = getauxval(AT_HWCAP3);
    has_ls64 = ((hwcap3 & HWCAP3_LS64) != 0);
    if(has_ls64 && COMPILED_LS64 == 0){
        gdr_warn("CPU supports LS64 but the current binary is not compiled with LS64 support. Disabling LS64 usage.\n");
        has_ls64 = 0;
    }
    gdr_dbg("ls64=%d neon=%d sve=%d\n", has_ls64, has_neon, has_sve);

    memcpy_uncached_store_16B = memcpy_uncached_store_arm64;
    memcpy_uncached_store_16B_name = "STP";
    memcpy_uncached_store_32B = memcpy_uncached_store_arm64;
    memcpy_uncached_store_32B_name = "STP";
    memcpy_uncached_load_16B = memcpy_uncached_load_arm64;
    memcpy_uncached_load_16B_name = "LDP";
    memcpy_uncached_load_32B = memcpy_uncached_load_arm64;
    memcpy_uncached_load_32B_name = "LDP";
    if (has_ls64) {
        memcpy_uncached_store_64B = memcpy_uncached_store_ls64;
        memcpy_uncached_store_64B_name = "ST64B";
        memcpy_uncached_load_64B = memcpy_uncached_load_ls64;
        memcpy_uncached_load_64B_name = "LD64B";
    }
    else if (has_neon) {
        memcpy_uncached_store_64B = memcpy_uncached_store_neon;
        memcpy_uncached_store_64B_name = "NEON";
        memcpy_uncached_load_64B = memcpy_uncached_load_neon;
        memcpy_uncached_load_64B_name = "NEON";
    }

#endif // GDRAPI_ARM64

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

static int aligned_copy(char *dest, const char *src, size_t size, size_t offset_addr, bool device_is_dest){
    while (size) {
        // Find the least significant set bit
        size_t access_width = offset_addr & -offset_addr;
        if(offset_addr == 0)
            access_width = 8;
        while(access_width > size)
            access_width >>= 1;
        switch(access_width){
            case 1:
                device_is_dest ? (WRITE_ONCE(*(uint8_t *)dest, *(uint8_t *)src)) : (*(uint8_t *)dest = READ_ONCE(*(uint8_t *)src));
                break;
            case 2:
                device_is_dest ? (WRITE_ONCE(*(uint16_t *)dest, *(uint16_t *)src)) : (*(uint16_t *)dest = READ_ONCE(*(uint16_t *)src));
                break;
            case 4:
                device_is_dest ? (WRITE_ONCE(*(uint32_t *)dest, *(uint32_t *)src)) : (*(uint32_t *)dest = READ_ONCE(*(uint32_t *)src));
                break;
            case 8:
                device_is_dest ? (WRITE_ONCE(*(uint64_t *)dest, *(uint64_t *)src)) : (*(uint64_t *)dest = READ_ONCE(*(uint64_t *)src));
                break;
            default:
                assert(false);
                break;
        }
        dest += access_width;
        src += access_width;
        size -= access_width;
        offset_addr = (offset_addr + access_width) & 0xF;
    }
    return 0;
}

static int gdr_copy_to_mapping_internal(void *map_d_ptr, const void *h_ptr, size_t size, gdr_mapping_type_t mapping_type)
{
    const int wc_mapping = (mapping_type == GDR_MAPPING_TYPE_WC);
    const int device_mapping = (mapping_type == GDR_MAPPING_TYPE_DEVICE);
    do {
        if(wc_mapping){
#if defined(GDRAPI_X86) || defined(GDRAPI_ARM64)
            // Assumption: always perform aligned accesses to device, so always align map_d_ptr first
            size_t offset_addr = (uintptr_t) map_d_ptr & 0xF;
            if(size < 16){
                aligned_copy(map_d_ptr, h_ptr, size, offset_addr, true);
                break;
            } else if(offset_addr > 0){
                aligned_copy(map_d_ptr, h_ptr, 16 - offset_addr, offset_addr, true);
                map_d_ptr += 16 - offset_addr;
                h_ptr += 16 - offset_addr;
                size -= 16 - offset_addr;
            }
            if(size >= 16 && ptr_is_aligned(map_d_ptr, 16) && !ptr_is_aligned(map_d_ptr, 32)){
                if (memcpy_uncached_store_16B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_to_mapping once\n", memcpy_uncached_store_16B_name);
                    memcpy_uncached_store_16B(map_d_ptr, h_ptr, 16);
                    map_d_ptr += 16;
                    h_ptr += 16;
                    size -= 16;
                }
            }
            if(size >= 32 && ptr_is_aligned(map_d_ptr, 32) && !ptr_is_aligned(map_d_ptr, 64)){
                if (memcpy_uncached_store_32B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_to_mapping once\n", memcpy_uncached_store_32B_name);
                    memcpy_uncached_store_32B(map_d_ptr, h_ptr, 32);
                    map_d_ptr += 32;
                    h_ptr += 32;
                    size -= 32;
                }
            }
            if(size >= 64 && ptr_is_aligned(map_d_ptr, 64)){
                size_t chunk = size & ~63ULL;
                if (memcpy_uncached_store_64B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_to_mapping\n", memcpy_uncached_store_64B_name);
                    memcpy_uncached_store_64B(map_d_ptr, h_ptr, chunk);
                    map_d_ptr += chunk;
                    h_ptr += chunk;
                    size -= chunk;
                }
            }
            if(size >= 32 && ptr_is_aligned(map_d_ptr, 32)){
                size_t chunk = size & ~31ULL;
                if (memcpy_uncached_store_32B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_to_mapping\n", memcpy_uncached_store_32B_name);
                    memcpy_uncached_store_32B(map_d_ptr, h_ptr, chunk);
                    map_d_ptr += chunk;
                    h_ptr += chunk;
                    size -= chunk;
                }
            }
            if(size >= 16 && ptr_is_aligned(map_d_ptr, 16)){
                size_t chunk = size & ~15ULL;
                if (memcpy_uncached_store_16B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_to_mapping\n", memcpy_uncached_store_16B_name);
                    memcpy_uncached_store_16B(map_d_ptr, h_ptr, chunk);
                    map_d_ptr += chunk;
                    h_ptr += chunk;
                    size -= chunk;
                }
            }
            if(size > 0){
                assert(ptr_is_aligned(map_d_ptr, 16));
                aligned_copy(map_d_ptr, h_ptr, size, 0, true);
            }
            break;
#endif
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

    if (gdr_has_mix_mapping()) {
        // All combinations of (ld, st) followed by (ld, st) targetting the
        // same CUDA buffer is possible. When using multiple mapping types
        // targeting the same buffer, we need memory fence to guarantee the
        // program order.
        memory_fence();
    } else if (wc_mapping) {
        // fencing is needed even for plain memcpy(), due to performance
        // being hit by delayed flushing of WC buffers
        wc_store_fence();
    }

    return 0;
}

static int gdr_copy_from_mapping_internal(void *h_ptr, const void *map_d_ptr, size_t size, gdr_mapping_type_t mapping_type)
{
    const int wc_mapping = (mapping_type == GDR_MAPPING_TYPE_WC);
    const int device_mapping = (mapping_type == GDR_MAPPING_TYPE_DEVICE);

    do {
        if(wc_mapping){
#if defined(GDRAPI_X86) || defined(GDRAPI_ARM64)
            // Assumption: always perform aligned accesses to device, so always align map_d_ptr first
            size_t offset_addr = (uintptr_t) map_d_ptr & 0xF;
            if(size < 16){
                aligned_copy(h_ptr, map_d_ptr, size, offset_addr, false);
                break;
            } else if(offset_addr > 0){
                aligned_copy(h_ptr, map_d_ptr, 16 - offset_addr, offset_addr, false);
                map_d_ptr += 16 - offset_addr;
                h_ptr += 16 - offset_addr;
                size -= 16 - offset_addr;
            }
            if(size >= 16 && ptr_is_aligned(map_d_ptr, 16) && !ptr_is_aligned(map_d_ptr, 32)){
                if (memcpy_uncached_load_16B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_from_mapping once\n", memcpy_uncached_load_16B_name);
                    memcpy_uncached_load_16B(h_ptr, map_d_ptr, 16);
                    map_d_ptr += 16;
                    h_ptr += 16;
                    size -= 16;
                }
            }
            if(size >= 32 && ptr_is_aligned(map_d_ptr, 32) && !ptr_is_aligned(map_d_ptr, 64)){
                if (memcpy_uncached_load_32B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_from_mapping once\n", memcpy_uncached_load_32B_name);
                    memcpy_uncached_load_32B(h_ptr, map_d_ptr, 32);
                    map_d_ptr += 32;
                    h_ptr += 32;
                    size -= 32;
                }
            }
            if(size >= 64 && ptr_is_aligned(map_d_ptr, 64)){
                size_t chunk = size & ~63ULL;
                if (memcpy_uncached_load_64B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_from_mapping\n", memcpy_uncached_load_64B_name);
                    memcpy_uncached_load_64B(h_ptr, map_d_ptr, chunk);
                    map_d_ptr += chunk;
                    h_ptr += chunk;
                    size -= chunk;
                }
            }
            if(size >= 32 && ptr_is_aligned(map_d_ptr, 32)){
                size_t chunk = size & ~31ULL;
                if (memcpy_uncached_load_32B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_from_mapping\n", memcpy_uncached_load_32B_name);
                    memcpy_uncached_load_32B(h_ptr, map_d_ptr, chunk);
                    map_d_ptr += chunk;
                    h_ptr += chunk;
                    size -= chunk;
                }
            }
            if(size >= 16 && ptr_is_aligned(map_d_ptr, 16)){
                size_t chunk = size & ~15ULL;
                if (memcpy_uncached_load_16B) {
                    gdr_dbgc(1, "using %s implementation of gdr_copy_from_mapping\n", memcpy_uncached_load_16B_name);
                    memcpy_uncached_load_16B(h_ptr, map_d_ptr, chunk);
                    map_d_ptr += chunk;
                    h_ptr += chunk;
                    size -= chunk;
                }
            }
            if(size > 0){
                assert(ptr_is_aligned(map_d_ptr, 16));
                aligned_copy(h_ptr, map_d_ptr, size, 0, false);
            }
            break;
#endif
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

    if (gdr_has_mix_mapping()) {
        // All combinations of (ld, st) followed by (ld, st) targetting the
        // same CUDA buffer is possible. When using multiple mapping types
        // targeting the same buffer, we need memory fence to guarantee the
        // program order.
        memory_fence();
    } else if (wc_mapping) {
        // fencing because of NT stores
        // potential optimization: issue only when NT stores are actually emitted
        // ARM always requires a fence because of weak ordering model
        wc_store_fence();
    }
    
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

int gdr_get_attribute(gdr_t g, gdr_attr_t attr, int *v)
{
    int ret = 0;
    int retcode;
    *v = 0;
    struct GDRDRV_IOC_GET_ATTR_PARAMS params;

    // check attribute first
    switch (attr) {
        case GDR_ATTR_USE_PERSISTENT_MAPPING:
            if (g->cache_backend == GDR_USE_DMABUF) {
                *v = 1;
                goto out;
            }
            params.attr = GDRDRV_ATTR_USE_PERSISTENT_MAPPING;
            break;
        case GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE:
            if (g->cache_backend == GDR_USE_DMABUF) {
                *v = 1;
                goto out;
            }
            params.attr = GDRDRV_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE;
            break;
        case GDR_ATTR_USING_DMA_BUF_MMAP:
            if (g->cache_backend == GDR_USE_DMABUF) {
                *v = 1;
            }
            goto out;
        default:
            gdr_dbg("undefined attribute\n");
            ret = EINVAL;
            goto out;
    }

    // If gdrdrv does not support attribute querying, assume that the value is 0.
    if (g->gdrdrv_version < GDRDRV_MINIMUM_VERSION_WITH_GET_ATTR) {
        gdr_dbg("gdrdrv is too old and does not support querying attributes\n");
        *v = 0;
        goto out;
    }

    retcode = ioctl(g->fd, GDRDRV_IOC_GET_ATTR, &params);
    if (-EINVAL == retcode) {
        // gdrdrv might be too old to query this attr.
        // Assume 0.
        *v = 0;
        goto out;
    } else if (0 != retcode) {
        ret = errno;
        gdr_err("ioctl error (errno=%d)\n", ret);
        goto out;
    }

    *v = params.val;

out:
    return ret;
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
    params.handle = mh->backend.gdrdrv_memh.handle;

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

int gdr_get_mapping_type_string(gdr_mapping_type_t mapping_type, const char **pstr)
{
    switch (mapping_type) {
        case GDR_MAPPING_TYPE_NONE:
            *pstr = "None";
            break;
        case GDR_MAPPING_TYPE_WC:
            *pstr = "Write Combining";
            break;
        case GDR_MAPPING_TYPE_CACHING:
            *pstr = "Caching";
            break;
        case GDR_MAPPING_TYPE_DEVICE:
            *pstr = "Device";
            break;
        default:
            return EINVAL;
    }
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
