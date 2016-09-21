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

#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"


int main(int argc, char *argv[])
{
    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    const size_t _size = 256*1024+16; //32*1024+8;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    printf("buffer size: %zu\n", size);
    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    //OUT << "device ptr: " << hex << d_A << dec << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    uint32_t *init_buf = new uint32_t[size];
    uint32_t *copy_buf = new uint32_t[size];

    init_hbuf_walking_bit(init_buf, size);
    memset(copy_buf, 0, sizeof(*copy_buf) * sizeof(uint32_t));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;
    BEGIN_CHECK {
        CUdeviceptr d_ptr = d_A;

        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        BREAK_IF_NEQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, 0U);

        void *bar_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);
        //OUT << "bar_ptr: " << bar_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        int off = d_ptr - info.va;
        cout << "off: " << off << endl;

        uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
        //OUT << "buf_ptr:" << buf_ptr << endl;

        printf("check 1: MMIO CPU initialization + read back via cuMemcpy D->H\n");
        init_hbuf_walking_bit(buf_ptr, size);
        //mmiowcwb();
        ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);

        printf("check 2: gdr_copy_to_bar() + read back via cuMemcpy D->H\n");
        gdr_copy_to_bar(buf_ptr, init_buf, size);
        ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        fflush(stdout);

        printf("check 3: gdr_copy_to_bar() + read back via gdr_copy_from_bar()\n");
        gdr_copy_to_bar(buf_ptr, init_buf, size);
        gdr_copy_from_bar(copy_buf, buf_ptr, size);
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);

        int extra_dwords = 5;
        int extra_off = extra_dwords * sizeof(uint32_t);
        printf("check 4: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d dwords offset\n", extra_dwords);
        gdr_copy_to_bar(buf_ptr + extra_dwords, init_buf, size - extra_off);
        gdr_copy_from_bar(copy_buf, buf_ptr + extra_dwords, size - extra_off);
        compare_buf(init_buf, copy_buf, size - extra_off);

        extra_off = 11;
        printf("check 4: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d bytes offset\n", extra_off);
        gdr_copy_to_bar((char*)buf_ptr + extra_off, init_buf, size);
        gdr_copy_from_bar(copy_buf, (char*)buf_ptr + extra_off, size);
        compare_buf(init_buf, copy_buf, size);

        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFree(d_A));
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
