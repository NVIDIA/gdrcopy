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

#pragma once

#include <stdarg.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda.h>
#include <check.h>
#include <map>
#include <gdrapi.h>

/**
 * Memory barrier
 */
#if defined(GDRAPI_X86)
#define MB() asm volatile("mfence":::"memory")
#elif defined(GDRAPI_POWER)
#define MB() asm volatile("sync":::"memory")
#else
#define MB() asm volatile("":::"memory")
#endif

/**
 * Clock used for timing
 */
//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC


#define EXIT_WAIVED 2

#define BEGIN_GDRCOPY_TEST(__testname)                                  \
START_TEST(__testname) {                                                \
    testname = #__testname;                                             \
    print_dbg("&&&& RUNNING " # __testname "\n");                       \
    fflush(stdout);                                                     \
    fflush(stderr);                                                     \
    pid_t __pid = fork();                                               \
    if (__pid < 0) {                                                    \
        print_dbg("Cannot fork\n");                                     \
        print_dbg("&&&& FAILED " # __testname "\n");                    \
        ck_abort();                                                     \
    }                                                                   \
    if (__pid == 0) {

#define END_GDRCOPY_TEST }                                              \
    if (__pid > 0) {                                                    \
        int __child_exit_status = -EINVAL;                              \
        if (waitpid(__pid, &__child_exit_status, 0) == -1) {            \
            print_dbg("waitpid returned an error\n");                   \
            print_dbg("&&&& FAILED %s\n", gdrcopy::test::testname);     \
            ck_abort();                                                 \
        }                                                               \
        if (__child_exit_status == EXIT_SUCCESS)                        \
            print_dbg("&&&& PASSED %s\n", gdrcopy::test::testname);     \
        else if (__child_exit_status == EXIT_WAIVED)                    \
            print_dbg("&&&& WAIVED %s\n", gdrcopy::test::testname);     \
        else {                                                          \
            print_dbg("&&&& FAILED %s\n", gdrcopy::test::testname);     \
            ck_abort();                                                 \
        }                                                               \
    }                                                                   \
} END_TEST

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

#define ASSERT_EQ(P, V) ASSERT((P) == (V))
#define CHECK_EQ(P, V) ASSERT((P) == (V))
#define ASSERT_NEQ(P, V) ASSERT(!((P) == (V)))
#define BREAK_IF_NEQ(P, V) if((P) != (V)) break
#define BEGIN_CHECK do
#define END_CHECK while(0)



namespace gdrcopy {
    namespace test {
        extern std::map<CUdeviceptr, CUdeviceptr> _allocations;

        static inline gdr_t gdr_open_safe()
        {
            gdr_t g = gdr_open();
            if (!g) {
                fprintf(stderr, "gdr_open error: Is gdrdrv driver installed and loaded?\n");
                exit(EXIT_FAILURE);
            }
            return g;
        }

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

        extern bool print_dbg_msg;

        void print_dbg(const char* fmt, ...);

        extern const char *testname;

        static inline bool operator==(const gdr_mh_t &a, const gdr_mh_t &b) {
            return a.h == b.h;
        }

        static const gdr_mh_t null_mh = {0};

        int compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size);

        void init_hbuf_walking_bit(uint32_t *h_buf, size_t size);

        void init_hbuf_linear_ramp(uint32_t *h_buf, size_t size);

        bool check_gdr_support(CUdevice dev);

        void print_histogram(double *lat_arr, int count, int *bin_arr, int num_bins, double min, double max);
    }
}
