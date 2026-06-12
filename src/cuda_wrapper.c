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

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "cuda_wrapper.h"
#include "gdrapi_internal.h"

#define GDR_CUDA_LIB_NAME "libcuda.so"
#define GDR_CUDA_DMABUF_MMAP_MIN_VERSION 13030

#define MAX(a,b) \
({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
  _a > _b ? _a : _b; })

typedef struct {
    void **func_ptr;
    const char *name;
} gdr_cuda_symbol_t;

struct gdr_cuda_function_table {
    void *handle;
    void *cuGetProcAddress;
};

static struct gdr_cuda_function_table *gdr_cuda_ftable = NULL;
static int gdr_cuda_ftable_refcount = 0;

CUresult (*gdr_cuInit)(unsigned int) = NULL;
CUresult (*gdr_cuDeviceGet)(CUdevice *device, int ordinal) = NULL;
CUresult (*gdr_cuDeviceGetCount)(int *count) = NULL;
CUresult (*gdr_cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev) = NULL;
CUresult (*gdr_cuMemGetHandleForAddressRange)(void *handle, CUdeviceptr ptr, size_t size, CUmemRangeHandleType type, unsigned long long flags) = NULL;
CUresult (*gdr_cuCtxGetCurrent)(CUcontext *pctx) = NULL;
CUresult (*gdr_cuCtxSetCurrent)(CUcontext ctx) = NULL;
CUresult (*gdr_cuDevicePrimaryCtxRetain)(CUcontext *pctx, CUdevice dev) = NULL;
CUresult (*gdr_cuDevicePrimaryCtxRelease)(CUdevice dev) = NULL;
CUresult (*gdr_cuPointerGetAttribute)(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) = NULL;
CUresult (*gdr_cuGetErrorString)(CUresult error, const char **pStr) = NULL;
CUresult (*gdr_cuGetErrorName)(CUresult error, const char **pStr) = NULL;
typedef CUresult (*cuGetProcAddress_t)(const char* symbol, void** pfn, int cudaVersion, uint64_t flags, CUdriverProcAddressQueryResult* symbolStatus);

static const gdr_cuda_symbol_t gdr_cuda_symbols[] = {
    { (void**)&gdr_cuInit,                        "cuInit",                         },
    { (void**)&gdr_cuDeviceGet,                   "cuDeviceGet",                    },
    { (void**)&gdr_cuDeviceGetCount,              "cuDeviceGetCount",               },
    { (void**)&gdr_cuDeviceGetAttribute,          "cuDeviceGetAttribute",           },
    { (void**)&gdr_cuMemGetHandleForAddressRange, "cuMemGetHandleForAddressRange",  },
    { (void**)&gdr_cuCtxGetCurrent,              "cuCtxGetCurrent",                 },
    { (void**)&gdr_cuCtxSetCurrent,              "cuCtxSetCurrent",                 },
    { (void**)&gdr_cuDevicePrimaryCtxRetain,     "cuDevicePrimaryCtxRetain",        },
    { (void**)&gdr_cuDevicePrimaryCtxRelease,    "cuDevicePrimaryCtxRelease",       },
    { (void**)&gdr_cuPointerGetAttribute,        "cuPointerGetAttribute",           },
    { (void**)&gdr_cuGetErrorString,             "cuGetErrorString",                 },
    { (void**)&gdr_cuGetErrorName,               "cuGetErrorName",                   },
};

#define GDR_CUDA_SYMBOL_COUNT (sizeof(gdr_cuda_symbols) / sizeof(gdr_cuda_symbols[0]))

static int gdr_cuda_load_function(void **func_ptr, const char *func_name, int driver_version) {
    CUresult res;

    if (!gdr_cuda_ftable || !gdr_cuda_ftable->cuGetProcAddress) {
        gdr_err("gdr_cuda_load_function: cuGetProcAddress not available\n");
        return -1;
    }

    res = ((cuGetProcAddress_t)gdr_cuda_ftable->cuGetProcAddress)(func_name, func_ptr, driver_version, CU_GET_PROC_ADDRESS_DEFAULT, NULL);
    if (res != CUDA_SUCCESS) {
        gdr_err("gdr_cuda_load_function: cuGetProcAddress failed for '%s' (error %u)\n", func_name, res);
        return res;
    }
    if (!*func_ptr) {
        gdr_err("gdr_cuda_load_function: '%s' not supported on this platform or for the requested driver version\n", func_name);
        return ENOTSUP;
    }

    return 0;
}

static int gdr_cuda_ftable_init(struct gdr_cuda_function_table **ftable) {
    int status = 0;
    void *handle = NULL;
    struct gdr_cuda_function_table *table = NULL;
    int driver_version = 0;

    handle = dlopen(GDR_CUDA_LIB_NAME, RTLD_LAZY);
    if (!handle) {
        gdr_err("Failed to open %s: %s\n", GDR_CUDA_LIB_NAME, dlerror());
        status = ENOENT;
        goto out;
    }

    table = (struct gdr_cuda_function_table *)malloc(sizeof(struct gdr_cuda_function_table));
    if (!table) {
        gdr_err("Failed to allocate memory for CUDA function table\n");
        status = ENOMEM;
        goto out;
    }

    CUresult (*gdr_cuDriverGetVersion)(int *driverVersion) = (CUresult (*)(int *))dlsym(handle, "cuDriverGetVersion");
    if (!gdr_cuDriverGetVersion) {
        gdr_err("Failed to resolve cuDriverGetVersion: %s\n", dlerror());
        status = ENOENT;
        goto out;
    }
    CUresult res = gdr_cuDriverGetVersion(&driver_version);
    if (res != CUDA_SUCCESS) {
        gdr_err("Failed to query CUDA driver version (error %u)\n", res);
        status = res;
        goto out;
    }

    if (driver_version < GDR_CUDA_DMABUF_MMAP_MIN_VERSION) {
        gdr_err("CUDA 13.3 or newer required for dmabuf mmap backend(reported version %d)\n", driver_version);
        status = ENOTSUP;
        goto out;
    }

    table->cuGetProcAddress = dlsym(handle, "cuGetProcAddress");
    if (!table->cuGetProcAddress) {
        gdr_err("Failed to resolve cuGetProcAddress: %s\n", dlerror());
        status = ENOENT;
        goto out;
    }

    table->handle = handle;
    *ftable = table;

    for (size_t i = 0; i < GDR_CUDA_SYMBOL_COUNT; ++i) {
        int res = gdr_cuda_load_function(gdr_cuda_symbols[i].func_ptr,
                                         gdr_cuda_symbols[i].name,
                                         GDR_CUDA_DMABUF_MMAP_MIN_VERSION);
        if (res != 0) {
            *(gdr_cuda_symbols[i].func_ptr) = NULL;
            gdr_err("Error: CUDA function '%s' not available in this driver version\n",
                    gdr_cuda_symbols[i].name);
            status = res;
            goto out;
        }
    }

out:
    if (status != 0) {
        if (handle) {
            dlclose(handle);
        }
        if (table) {
            free(table);
        }
        for (size_t i = 0; i < GDR_CUDA_SYMBOL_COUNT; ++i) {
            *(gdr_cuda_symbols[i].func_ptr) = NULL;
        }
    }
    return status;
}

static void gdr_cuda_ftable_cleanup(struct gdr_cuda_function_table *ftable) {
    if (ftable) {
        if (ftable->handle) {
            dlclose(ftable->handle);
        }
        free(ftable);
    }
    for (size_t i = 0; i < GDR_CUDA_SYMBOL_COUNT; ++i) {
        *(gdr_cuda_symbols[i].func_ptr) = NULL;
    }
}

int gdr_cuda_init(void)
{
    if (gdr_cuda_ftable_refcount > 0) {
        gdr_cuda_ftable_refcount++;
        return 0;
    }

    int status = gdr_cuda_ftable_init(&gdr_cuda_ftable);
    if (status) {
        gdr_err("Error in gdr_cuda_ftable_init: %d\n", status);
        return status;
    }

    CUresult res = gdr_cuInit(0);
    if (res != CUDA_SUCCESS) {
        gdr_err("Error in gdr_cuInit: %d\n", res);
        status = res;
        goto out;
    }

    gdr_cuda_ftable_refcount = 1;

out:
    if (status) {
        gdr_cuda_ftable_cleanup(gdr_cuda_ftable);
        gdr_cuda_ftable = NULL;
    }
    return status;
}

int gdr_cuda_any_device_supports_dmabuf_mmap(int *supported)
{
    int device_count = 0;
    CUresult res;
    int status = 0;

    if (gdr_cuda_ftable_refcount == 0) {
        *supported = 0;
        gdr_err("gdr_cuda_any_device_supports_dmabuf_mmap: gdr_cuda not initialized, initialize using gdr_cuda_init()\n");
        return EINVAL;
    }

    *supported = 0;
    res = gdr_cuDeviceGetCount(&device_count);
    if (res != CUDA_SUCCESS) {
        gdr_err("Error in gdr_cuDeviceGetCount: %d\n", res);
        status = res;
        goto out;
    }

    for (int i = 0; i < device_count; i++) {
        CUdevice dev;
        int dmabuf_mmap_supported = 0;

        res = gdr_cuDeviceGet(&dev, i);
        if (res != CUDA_SUCCESS) {
            gdr_err("Error in gdr_cuDeviceGet(%d): %d\n", i, res);
            status = res;
            goto out;
        }
        status = gdr_cuda_device_supports_dmabuf_mmap(dev, &dmabuf_mmap_supported);
        if (status != 0) {
            goto out;
        }
        if (dmabuf_mmap_supported) {
            *supported = 1;
            return 0;
        }
    }

out:
    return status;
}

int gdr_cuda_device_supports_dmabuf_mmap(CUdevice dev, int *supported)
{
    CUresult res;

    if (gdr_cuda_ftable_refcount == 0) {
        *supported = 0;
        gdr_err("gdr_cuda_device_supports_dmabuf_mmap: gdr_cuda not initialized, initialize using gdr_cuda_init()\n");
        return EINVAL;
    }

    *supported = 0;
    res = gdr_cuDeviceGetAttribute(supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
    if (res != CUDA_SUCCESS) {
        gdr_err("Error in gdr_cuDeviceGetAttribute(DMA_BUF_MMAP_SUPPORTED): %d\n", res);
        return res;
    }

    return 0;
}

int gdr_cuda_device_is_coherent(CUdevice dev, int *coherent)
{
    CUresult res;

    if (gdr_cuda_ftable_refcount == 0) {
        *coherent = 0;
        gdr_err("gdr_cuda_device_is_coherent: gdr_cuda not initialized, initialize using gdr_cuda_init()\n");
        return EINVAL;
    }

    *coherent = 0;
    res = gdr_cuDeviceGetAttribute(coherent,
                                   CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
                                   dev);
    if (res != CUDA_SUCCESS) {
        gdr_err("Error in gdr_cuDeviceGetAttribute(PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES): %d\n", res);
        return res;
    }

    return 0;
}

void gdr_cuda_cleanup(void) {
    if (gdr_cuda_ftable_refcount == 1) {
        assert(gdr_cuda_ftable != NULL);
        gdr_cuda_ftable_cleanup(gdr_cuda_ftable);
        gdr_cuda_ftable = NULL;
    }
    gdr_cuda_ftable_refcount = MAX(gdr_cuda_ftable_refcount - 1, 0);
}
