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

#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError_enum {
    CUDA_SUCCESS                          = 0,
    CU_RESULT_MAX                         = 0x7FFFFFFF
} CUresult;

typedef enum CUmemRangeHandleType_enum {
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD   = 0x1,
    CU_MEM_RANGE_HANDLE_TYPE_MAX          = 0x7FFFFFFF
} CUmemRangeHandleType;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL   = 9,
    CU_POINTER_ATTRIBUTE_MAX              = 0x7FFFFFFF
} CUpointer_attribute;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_MAX               = 0x7FFFFFFF
} CUdevice_attribute;

typedef enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS           = 0,
    CU_GET_PROC_ADDRESS_QUERY_RESULT_MAX  = 0x7FFFFFFF
} CUdriverProcAddressQueryResult;

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;

#define CU_GET_PROC_ADDRESS_DEFAULT                   0
#define CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE   0x1

int gdr_cuda_init(void);
int gdr_cuda_any_device_supports_dmabuf_mmap(int *supported);
int gdr_cuda_device_supports_dmabuf_mmap(CUdevice dev, int *supported);
int gdr_cuda_device_is_coherent(CUdevice dev, int *coherent);
void gdr_cuda_cleanup(void);
extern CUresult (*gdr_cuMemGetHandleForAddressRange)(void *handle, CUdeviceptr ptr, size_t size, CUmemRangeHandleType type, unsigned long long flags);
extern CUresult (*gdr_cuCtxGetCurrent)(CUcontext *pctx);
extern CUresult (*gdr_cuCtxSetCurrent)(CUcontext ctx);
extern CUresult (*gdr_cuDeviceGet)(CUdevice *device, int ordinal);
extern CUresult (*gdr_cuDevicePrimaryCtxRetain)(CUcontext *pctx, CUdevice dev);
extern CUresult (*gdr_cuDevicePrimaryCtxRelease)(CUdevice dev);
extern CUresult (*gdr_cuPointerGetAttribute)(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
extern CUresult (*gdr_cuGetErrorString)(CUresult error, const char **pStr);
extern CUresult (*gdr_cuGetErrorName)(CUresult error, const char **pStr);

/* On CUDA error, log, assign the CUresult to retcode_var, and goto label. */
#define CUCHECK_GOTO(stmt, retcode_var, label)                              \
    do {                                                                    \
        CUresult _cuck_r = (stmt);                                          \
        if (_cuck_r != CUDA_SUCCESS) {                                      \
            const char *_cuck_es = NULL, *_cuck_en = NULL;                  \
            gdr_cuGetErrorString(_cuck_r, &_cuck_es);                       \
            gdr_cuGetErrorName(_cuck_r, &_cuck_en);                         \
            gdr_err("\"%s\" failed at %s:%d (%d: %s: \"%s\")\n",            \
                    #stmt, __FILE__, __LINE__, (int)_cuck_r,                \
                    _cuck_en ? _cuck_en : "?",                              \
                    _cuck_es ? _cuck_es : "?");                             \
            (retcode_var) = _cuck_r;                                        \
            goto label;                                                     \
        }                                                                   \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* CUDA_WRAPPER_H */