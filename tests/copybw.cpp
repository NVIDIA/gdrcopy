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

#include <stdlib.h>
#include <getopt.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

using namespace gdrcopy::test;

// manually tuned...
int num_write_iters = 10000;
int num_read_iters  = 100;
size_t _size = 128*1024;
size_t copy_size = 0;
size_t copy_offset = 0;
int dev_id = 0;

void print_usage(const char *path)
{
    cout << "Usage: " << path << " [-h][-s <size>][-c <size>][-o <offset>][-d <gpu>][-w <iters>][-r <iters>][-a <fn>]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h              Print this help text" << endl;
    cout << "   -s <size>       Buffer allocation size (default: " << _size << ")" << endl;
    cout << "   -c <size>       Copy size (default: " << copy_size << ")" << endl;
    cout << "   -o <offset>     Copy offset (default: " << copy_offset << ")" << endl;
    cout << "   -d <gpu>        GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -w <iters>      Number of write iterations (default: " << num_write_iters << ")" << endl;
    cout << "   -r <iters>      Number of read iterations (default: " << num_read_iters << ")" << endl;
    cout << "   -a <fn>         GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                       Choices: cuMemAlloc, cuMemCreate" << endl;
}

void run_test(CUdeviceptr d_A, size_t size)
{
    uint32_t *init_buf = NULL;
    init_buf = (uint32_t *)malloc(size);
    ASSERT_NEQ(init_buf, (void*)0);
    init_hbuf_walking_bit(init_buf, size);

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, size), 0);
        cout << "map_d_ptr: " << map_d_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        cout << "info.va: " << hex << info.va << dec << endl;
        cout << "info.mapped_size: " << info.mapped_size << endl;
        cout << "info.page_size: " << info.page_size << endl;
        cout << "info.mapped: " << info.mapped << endl;
        cout << "info.wc_mapping: " << info.wc_mapping << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = info.va - d_A;
        cout << "page offset: " << off << endl;

        uint32_t *buf_ptr = (uint32_t *)((char *)map_d_ptr + off);
        cout << "user-space pointer:" << buf_ptr << endl;

        // copy to GPU benchmark
        cout << "writing test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_write_iters << endl;
        struct timespec beg, end;
        clock_gettime(MYCLOCK, &beg);
        for (int iter=0; iter<num_write_iters; ++iter)
            gdr_copy_to_mapping(mh, buf_ptr + copy_offset/4, init_buf, copy_size);
        clock_gettime(MYCLOCK, &end);

        double woMBps;
        {
            double byte_count = (double) copy_size * num_write_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            woMBps = Bps / 1024.0 / 1024.0;
            cout << "write BW: " << woMBps << "MB/s" << endl;
        }

        compare_buf(init_buf, buf_ptr + copy_offset/4, copy_size);

        // copy from GPU benchmark
        cout << "reading test, size=" << copy_size << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
        clock_gettime(MYCLOCK, &beg);
        for (int iter=0; iter<num_read_iters; ++iter)
            gdr_copy_from_mapping(mh, init_buf, buf_ptr + copy_offset/4, copy_size);
        clock_gettime(MYCLOCK, &end);

        double roMBps;
        {
            double byte_count = (double) copy_size * num_read_iters;
            double dt_ms = (end.tv_nsec-beg.tv_nsec)/1000000.0 + (end.tv_sec-beg.tv_sec)*1000.0;
            double Bps = byte_count / dt_ms * 1e3;
            roMBps = Bps / 1024.0 / 1024.0;
            cout << "read BW: " << roMBps << "MB/s" << endl;
        }

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);
}

int main(int argc, char *argv[])
{
    gpu_memalloc_fn_t galloc_fn = gpu_mem_alloc;
    gpu_memfree_fn_t gfree_fn = gpu_mem_free;

    while(1) {        
        int c;
        c = getopt(argc, argv, "s:d:o:c:w:r:a:h");
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
        case 'w':
            num_write_iters = strtol(optarg, NULL, 0);
            break;
        case 'r':
            num_read_iters = strtol(optarg, NULL, 0);
            break;
        case 'a':
            if (strcmp(optarg, "cuMemAlloc") == 0) {
                galloc_fn = gpu_mem_alloc;
                gfree_fn = gpu_mem_free;
            }
            else if (strcmp(optarg, "cuMemCreate") == 0) {
                galloc_fn = gpu_vmm_alloc;
                gfree_fn = gpu_vmm_free;
            }
            else {
                cerr << "Unrecognized fn argument" << endl;
                exit(EXIT_FAILURE);
            }
            break;
        case 'h':
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        default:
            fprintf(stderr, "ERROR: invalid option\n");
            exit(EXIT_FAILURE);
        }
    }
    
    if (!copy_size)
        copy_size = _size;

    if (copy_offset % sizeof(uint32_t) != 0) {
        fprintf(stderr, "ERROR: offset must be multiple of 4 bytes\n");
        exit(EXIT_FAILURE);
    }

    if (copy_offset + copy_size > _size) {
        fprintf(stderr, "ERROR: offset + copy size run past the end of the buffer\n");
        exit(EXIT_FAILURE);
    }

    size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    ASSERTDRV(cuInit(0));

    int n_devices = 0;
    ASSERTDRV(cuDeviceGetCount(&n_devices));

    CUdevice dev;
    for (int n=0; n<n_devices; ++n) {
        
        char dev_name[256];
        int dev_pci_domain_id;
        int dev_pci_bus_id;
        int dev_pci_device_id;

        ASSERTDRV(cuDeviceGet(&dev, n));
        ASSERTDRV(cuDeviceGetName(dev_name, sizeof(dev_name) / sizeof(dev_name[0]), dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
        ASSERTDRV(cuDeviceGetAttribute(&dev_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));

        cout << "GPU id:" << n << "; name: " << dev_name 
            << "; Bus id: "
            << std::hex 
            << std::setfill('0') << std::setw(4) << dev_pci_domain_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id
            << ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id
            << std::dec
            << endl;
    }
    cout << "selecting device " << dev_id << endl;
    ASSERTDRV(cuDeviceGet(&dev, dev_id));


    CUcontext dev_ctx;
    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));

    cout << "testing size: " << _size << endl;
    cout << "rounded size: " << size << endl;

    ASSERT_EQ(check_gdr_support(dev), true);

    if (galloc_fn == gpu_mem_alloc)
        cout << "gpu alloc fn: cuMemAlloc" << endl;
    else
        cout << "gpu alloc fn: cuMemCreate" << endl;

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;
    cout << "device ptr: " << hex << d_A << dec << endl;

    run_test(d_A, size);

    ASSERTDRV(gfree_fn(&mhandle));

    ASSERTDRV(cuDevicePrimaryCtxRelease(dev));

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
