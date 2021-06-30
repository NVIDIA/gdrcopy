/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
#include <stdarg.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <cuda.h>
#include "common.hpp"

#define ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

namespace gdrcopy {
    namespace test {
        bool print_dbg_msg = false;
        const char *testname = "";

        void print_dbg(const char* fmt, ...)
        {
            if (print_dbg_msg) {
                va_list ap;
                va_start(ap, fmt);
                vfprintf(stderr, fmt, ap);
            }
        }

        CUresult gpu_mem_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
        {
            CUresult ret = CUDA_SUCCESS;
            CUdeviceptr ptr, out_ptr;
            size_t allocated_size;

            if (aligned_mapping)
                allocated_size = size + GPU_PAGE_SIZE - 1;
            else
                allocated_size = size;

            ret = cuMemAlloc(&ptr, allocated_size);
            if (ret != CUDA_SUCCESS)
                return ret;

            if (set_sync_memops) {
                unsigned int flag = 1;
                ret = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr);
                if (ret != CUDA_SUCCESS) {
                    cuMemFree(ptr);
                    return ret;
                }
            }

            if (aligned_mapping)
                out_ptr = ROUND_UP(ptr, GPU_PAGE_SIZE);
            else
                out_ptr = ptr;

            handle->ptr = out_ptr;
            handle->unaligned_ptr = ptr;
            handle->size = size;
            handle->allocated_size = allocated_size;

            return CUDA_SUCCESS;
        }

        CUresult gpu_mem_free(gpu_mem_handle_t *handle)
        {
            CUresult ret = CUDA_SUCCESS;
            CUdeviceptr ptr;

            ret = cuMemFree(handle->unaligned_ptr);
            if (ret == CUDA_SUCCESS)
                memset(handle, 0, sizeof(gpu_mem_handle_t));

            return ret;
        }

#if CUDA_VERSION >= 11000
        /**
         * Allocating GPU memory using VMM API.
         * VMM API is available since CUDA 10.2. However, the RDMA support is added in CUDA 11.0.
         * Our tests are not useful without RDMA support. So, we enable this VMM allocation from CUDA 11.0.
         */
        CUresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
        {
            CUresult ret = CUDA_SUCCESS;

            size_t granularity, gran;
            CUmemAllocationProp mprop;
            CUdevice gpu_dev;
            size_t rounded_size;
            CUdeviceptr ptr = 0;
            CUmemGenericAllocationHandle mem_handle = 0;
            bool is_mapped = false;

            int RDMASupported = 0;

            int version;

            ret = cuDriverGetVersion(&version);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuDriverGetVersion\n");
                goto out;
            }

            if (version < 11000) {
                print_dbg("VMM with RDMA is not supported in this CUDA version.\n");
                ret = CUDA_ERROR_NOT_SUPPORTED;
                goto out;
            }

            ret = cuCtxGetDevice(&gpu_dev);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuCtxGetDevice\n");
                goto out;
            }

            ret = cuDeviceGetAttribute(&RDMASupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, gpu_dev);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuDeviceGetAttribute\n");
                goto out;
            }

            if (!RDMASupported) {
                print_dbg("GPUDirect RDMA is not supported on this GPU.\n");
                ret = CUDA_ERROR_NOT_SUPPORTED;
                goto out;
            }

            memset(&mprop, 0, sizeof(CUmemAllocationProp));
            mprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            mprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            mprop.location.id = gpu_dev;
            mprop.allocFlags.gpuDirectRDMACapable = 1;

            ret = cuMemGetAllocationGranularity(&gran, &mprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemGetAllocationGranularity\n");
                goto out;
            }

            // In case gran is smaller than GPU_PAGE_SIZE
            granularity = ROUND_UP(gran, GPU_PAGE_SIZE);

            rounded_size = ROUND_UP(size, granularity);
            ret = cuMemAddressReserve(&ptr, rounded_size, granularity, 0, 0);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemAddressReserve\n");
                goto out;
            }

            ret = cuMemCreate(&mem_handle, rounded_size, &mprop, 0);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemCreate\n");
                goto out;
            }

            ret = cuMemMap(ptr, rounded_size, 0, mem_handle, 0);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemMap\n");
                goto out;
            }
            is_mapped = true;

            CUmemAccessDesc access;
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = gpu_dev;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            ret = cuMemSetAccess(ptr, rounded_size, &access, 1);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemSetAccess\n");
                goto out;
            }

            // cuMemAddressReserve always returns aligned ptr
            handle->ptr = ptr;
            handle->handle = mem_handle;
            handle->size = size;
            handle->allocated_size = rounded_size;

out:
            if (ret != CUDA_SUCCESS) {
                if (is_mapped)
                    cuMemUnmap(ptr, rounded_size);
                
                if (mem_handle)
                    cuMemRelease(mem_handle);
                
                if (ptr)
                    cuMemAddressFree(ptr, rounded_size);
            }
            return ret;
        }

        CUresult gpu_vmm_free(gpu_mem_handle_t *handle)
        {
            CUresult ret;

            if (!handle || !handle->ptr)
                return CUDA_ERROR_INVALID_VALUE;

            ret = cuMemUnmap(handle->ptr, handle->allocated_size);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemUnmap\n");
                return ret;
            }

            ret = cuMemRelease(handle->handle);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemRelease\n");
                return ret;
            }

            ret = cuMemAddressFree(handle->ptr, handle->allocated_size);
            if (ret != CUDA_SUCCESS) {
                print_dbg("error in cuMemAddressFree\n");
                return ret;
            }

            memset(handle, 0, sizeof(gpu_mem_handle_t));

            return CUDA_SUCCESS;
        }
#else
        /* VMM with RDMA is not available before CUDA 11.0 */
        CUresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
        {
            return CUDA_ERROR_NOT_SUPPORTED;
        }

        CUresult gpu_vmm_free(gpu_mem_handle_t *handle)
        {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
#endif

        int compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
        {
            int diff = 0;
            if (size % 4 != 0U) {
                print_dbg("warning: buffer size %zu is not dword aligned, ignoring trailing bytes\n", size);
                size -= (size % 4);
            }
            unsigned ndwords = size/sizeof(uint32_t);
            for(unsigned  w = 0; w < ndwords; ++w) {
                if (ref_buf[w] != buf[w]) {
                    if (!diff) {
                        printf("%10.10s %8.8s %8.8s\n", "word", "content", "expected");
                    }
                    if (diff < 10) {
                        printf("%10d %08x %08x\n", w, buf[w], ref_buf[w]);
                    }
                    ++diff;
                }
            }
            if (diff) {
                print_dbg("check error: %d different dwords out of %d\n", diff, ndwords);
            }
            return diff;
        }

        void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
        {
            uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
            unsigned w;
            ASSERT_NEQ(h_buf, (void*)0);
            ASSERT_EQ(size % 4, 0U);
            //OUT << "filling mem with walking bit " << endl;
            for(w = 0; w<size/sizeof(uint32_t); ++w)
                h_buf[w] = base_value ^ (1<< (w%32));
        }

        void init_hbuf_linear_ramp(uint32_t *h_buf, size_t size)
        {
            uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
            unsigned w;
            ASSERT_NEQ(h_buf, (void*)0);
            ASSERT_EQ(size % 4, 0U);
            //OUT << "filling mem with walking bit " << endl;
            for(w = 0; w<size/sizeof(uint32_t); ++w)
                h_buf[w] = w;
        }

        bool check_gdr_support(CUdevice dev)
        {
            #if CUDA_VERSION >= 11030
            int drv_version;
            ASSERTDRV(cuDriverGetVersion(&drv_version));

            // Starting from CUDA 11.3, CUDA provides an ability to check GPUDirect RDMA support.
            if (drv_version >= 11030) {
                int gdr_support = 0;
                ASSERTDRV(cuDeviceGetAttribute(&gdr_support, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev));

                if (!gdr_support)
                    print_dbg("This GPU does not support GPUDirect RDMA.\n");

                return !!gdr_support;
            }
            #endif

            // For older versions, we fall back to detect this support with gdr_pin_buffer.
            const size_t size = GPU_PAGE_SIZE;
            CUdeviceptr d_A;
            gpu_mem_handle_t mhandle;
            ASSERTDRV(gpu_mem_alloc(&mhandle, size, true, true));
            d_A = mhandle.ptr;

            gdr_t g = gdr_open_safe();

            gdr_mh_t mh;
            int status = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
            if (status != 0) {
                print_dbg("error in gdr_pin_buffer with code=%d\n", status);
                print_dbg("Your GPU might not support GPUDirect RDMA\n");
            }
            else
                ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

            ASSERT_EQ(gdr_close(g), 0);

            ASSERTDRV(gpu_mem_free(&mhandle));

            return status == 0;
        }

        void print_histogram(double *lat_arr, int count, int *bin_arr, int num_bins, double min, double max)
        {
            int den = (max - min) / num_bins;
            den = den > 0 ? den : 1;
            for (int j = 0; j < num_bins; j++) 
                bin_arr[j] = 0;
            for (int i = 0; i < count; i++) {
                bin_arr[(int) ((lat_arr[i] - min) / den)]++;
            }
            for (int j = 0; j < num_bins; j++) {
                printf("[%lf\t-\t%lf]\t%d\n", (min * (j + 1)), (min * (j + 2)), bin_arr[j]);
            }
        }
    }
}
