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
#include <getopt.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

#define OUT cout
//#define OUT TESTSTACK

//#define MYCLOCK CLOCK_REALTIME
//#define MYCLOCK CLOCK_RAW_MONOTONIC
#define MYCLOCK CLOCK_MONOTONIC

// manually tuned...
const int num_write_iters = 10000;
const int num_read_iters  = 100;

main(int argc, char *argv[])
{
    size_t _size = 128*1024;
    size_t copy_size = 0;
    size_t copy_offset = 0;
    int dev_id = 0;

    while(1) {        
        int c;
        c = getopt(argc, argv, "s:d:o:c:h");
        if (c == -1)
            break;

        switch (c) {
        case 's':
            _size = strtol(optarg, NULL, 0);
            break;
        case 'c':
            copy_size = strtol(optarg, NULL, 0);
            break;
        case 'o':
            copy_offset = strtol(optarg, NULL, 0);
            break;
        case 'd':
            dev_id = strtol(optarg, NULL, 0);
            break;
        case 'h':
            printf("syntax: %s -s <buf size> -c <copy size> -o <copy offset> -d <gpu dev id> -h\n");
            exit(EXIT_FAILURE);
            break;
        default:
            printf("ERROR: invalid option\n");
            exit(EXIT_FAILURE);
        }
    }
    
    if (!copy_size)
        copy_size = _size;

    if (copy_offset % sizeof(uint32_t) != 0) {
        printf("ERROR: offset must be multiple of 4 bytes\n");
        exit(EXIT_FAILURE);
    }

    if (copy_offset + copy_size > _size) {
        printf("ERROR: offset + copy size run past the end of the buffer\n");
        exit(EXIT_FAILURE);
    }

    size_t size = (_size + GDR_GPU_PAGE_SIZE - 1) & GDR_GPU_PAGE_MASK;

    int n_devices = 0;
    ASSERTRT(cudaGetDeviceCount(&n_devices));

    cudaDeviceProp prop;
    for (int n=0; n<n_devices; ++n) {
        ASSERTRT(cudaGetDeviceProperties(&prop,n));
        OUT << "GPU id:" << n << " name:" << prop.name 
            << " PCI domain: " << prop.pciDomainID 
            << " bus: " << prop.pciBusID 
            << " device: " << prop.pciDeviceID << endl;
    }
    OUT << "selecting device " << dev_id << endl;
    ASSERTRT(cudaSetDevice(dev_id));

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    OUT << "testing size: " << _size << endl;
    OUT << "rounded size: " << size << endl;

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    OUT << "device ptr: " << hex << d_A << dec << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    uint32_t *init_buf = NULL;
    init_buf = (uint32_t *)malloc(size);
    ASSERT_NEQ(init_buf, (void*)0);
    init_hbuf_walking_bit(init_buf, size);

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, 0U);

        void *bar_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);
        OUT << "bar_ptr: " << bar_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        OUT << "info.va: " << hex << info.va << dec << endl;
        OUT << "info.mapped_size: " << info.mapped_size << endl;
        OUT << "info.page_size: " << info.page_size << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = info.va - d_A;
        OUT << "page offset: " << off << endl;

        uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
        OUT << "user-space pointer:" << buf_ptr << endl;

        // copy to BAR benchmark
        cout << "BAR writing test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_write_iters << endl;
        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);
        for (int iter=0; iter<num_write_iters; ++iter)
            gdr_copy_to_bar(buf_ptr + copy_offset/4, init_buf, copy_size);
        clock_gettime(MYCLOCK, &end);

        double woMBps;
        {
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "BAR1 write BW: " << woMBps << "MB/s" << endl;
        }

        compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);

        // copy from BAR benchmark
        cout << "BAR reading test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
        clock_gettime(MYCLOCK, &beg);
        for (int iter=0; iter<num_read_iters; ++iter)
            gdr_copy_from_bar(init_buf, buf_ptr + copy_offset/4, copy_size);
        clock_gettime(MYCLOCK, &end);

        double roMBps;
        {
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "BAR1 read BW: " << roMBps << "MB/s" << endl;
        }

        OUT << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);

        OUT << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    OUT << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFree(d_A));
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
