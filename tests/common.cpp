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
#include <iostream>
#include <cuda.h>
#include "common.hpp"

namespace gdrcopy {
    namespace test {
        bool print_dbg_msg = false;
        std::map<CUdeviceptr, CUdeviceptr> _allocations;
        const char *testname = "";

        void print_dbg(const char* fmt, ...)
        {
            if (print_dbg_msg) {
                va_list ap;
                va_start(ap, fmt);
                vfprintf(stderr, fmt, ap);
            }
        }

        int compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
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
                    std::cerr << "This GPU does not support GPUDirect RDMA." << std::endl;

                return !!gdr_support;
            }
            #endif

            // For older versions, we fall back to detect this support with gdr_pin_buffer.
            const size_t size = GPU_PAGE_SIZE;
            CUdeviceptr d_A;
            ASSERTDRV(gpuMemAlloc(&d_A, size));

            gdr_t g = gdr_open_safe();

            gdr_mh_t mh;
            int status = gdr_pin_buffer(g, d_A, size, 0, 0, &mh);
            if (status != 0) {
                std::cerr << "error in gdr_pin_buffer with code=" << status << std::endl;
                std::cerr << "Your GPU might not support GPUDirect RDMA" << std::endl;
            }
            else
                ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

            ASSERT_EQ(gdr_close(g), 0);

            ASSERTDRV(gpuMemFree(d_A));

            return status == 0;
        }
    }
}
