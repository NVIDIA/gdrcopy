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

#pragma once

#include <map>
#include <cuda.h>

namespace gdrcopy {
    namespace test {
        static std::map<CUdeviceptr, CUdeviceptr> _allocations;

        static inline CUresult gpuMemAlloc(CUdeviceptr *pptr, size_t psize, bool align_to_gpu_page = true, bool set_sync_memops = true)
        {
            CUresult ret = CUDA_SUCCESS;
            CUdeviceptr ptr;
            size_t size;

            if (align_to_gpu_page)
                size = psize + GPU_PAGE_SIZE - 1;
            else
                size = psize;

            ret = cuMemAlloc(&ptr, size);
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

            if (align_to_gpu_page)
                *pptr = (ptr + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
            else
                *pptr = ptr;

            // Record the actual pointer for doing gpuMemFree later.
            _allocations[*pptr] = ptr;

            return CUDA_SUCCESS;
        }

        static inline CUresult gpuMemFree(CUdeviceptr pptr)
        {
            CUresult ret = CUDA_SUCCESS;
            CUdeviceptr ptr;

            if (_allocations.count(pptr) > 0) {
                ptr = _allocations[pptr];
                ret = cuMemFree(ptr);
                if (ret == CUDA_SUCCESS)
                    _allocations.erase(ptr);
                return ret;
            }
            else
                return CUDA_ERROR_INVALID_VALUE;
        }

        #define ASSERT(x)                                                       \
            do                                                                  \
                {                                                               \
                    if (!(x))                                                   \
                        {                                                       \
                            fprintf(stderr, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
                            exit(EXIT_FAILURE);                                 \
                        }                                                       \
                } while (0)

        #define ASSERTDRV(stmt)				\
            do                                          \
                {                                       \
                    CUresult result = (stmt);           \
                    if (result != CUDA_SUCCESS) {       \
                        const char *_err_name;          \
                        cuGetErrorName(result, &_err_name); \
                        fprintf(stderr, "CUDA error: %s\n", _err_name);   \
                    }                                   \
                    ASSERT(CUDA_SUCCESS == result);     \
                } while (0)

        #define ASSERTRT(stmt)				\
            do                                          \
                {                                       \
                    cudaError_t result = (stmt);           \
                    ASSERT(cudaSuccess == result);     \
                } while (0)

        static inline bool operator==(const gdr_mh_t &a, const gdr_mh_t &b) {
            return a.h == b.h;
        }

        static const gdr_mh_t null_mh = {0};

        #define ASSERT_EQ(P, V) ASSERT((P) == (V))
        #define CHECK_EQ(P, V) ASSERT((P) == (V))
        #define ASSERT_NEQ(P, V) ASSERT(!((P) == (V)))
        #define BREAK_IF_NEQ(P, V) if((P) != (V)) break
        #define BEGIN_CHECK do
        #define END_CHECK while(0)

        static int compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
        {
            int diff = 0;
            if (size % 4 != 0U) {
                printf("warning: buffer size %zu is not dword aligned, ignoring trailing bytes\n", size);
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
                printf("check error: %d different dwords out of %d\n", diff, ndwords);
            }
            return diff;
        }

        static void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
        {
            uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
            unsigned w;
            ASSERT_NEQ(h_buf, (void*)0);
            ASSERT_EQ(size % 4, 0U);
            //OUT << "filling mem with walking bit " << endl;
            for(w = 0; w<size/sizeof(uint32_t); ++w)
                h_buf[w] = base_value ^ (1<< (w%32));
        }

        static void init_hbuf_linear_ramp(uint32_t *h_buf, size_t size)
        {
            uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
            unsigned w;
            ASSERT_NEQ(h_buf, (void*)0);
            ASSERT_EQ(size % 4, 0U);
            //OUT << "filling mem with walking bit " << endl;
            for(w = 0; w<size/sizeof(uint32_t); ++w)
                h_buf[w] = w;
        }
    }
}
