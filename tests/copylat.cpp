/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda_runtime_api.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

using namespace gdrcopy::test;

// manually tuned...
int num_write_iters = 10000;
int num_read_iters = 100;
int small_size_iter_factor = 1000;
int warmup = 10;

int main(int argc, char *argv[])
{
    size_t _size = (size_t)1 << 24;
    size_t copy_size = 1;
    int dev_id = 0;
    bool do_cumemcpy = false;
    struct timespec beg, end;
    double lat_us;
    double bw;

    while(1) {        
        int c;
        c = getopt(argc, argv, "s:d:w:r:hcW:");
        if (c == -1)
            break;

        switch (c) {
            case 's':
                _size = strtol(optarg, NULL, 0);
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
            case 'W':
                warmup = strtol(optarg, NULL, 0);
                break;
            case 'c':
                do_cumemcpy = true;
                break;
            case 'h':
                printf("syntax: %s [-s <buf size>][-d <gpu dev id>][-w <write iters>][-r <read iters>][-h][-c][-w]\n"
                       "-c                   benchmark cuMemcpy\n"
                       "-w <# iterations>    modify warmup (default %d)\n",
                       argv[0],
                       warmup);
                exit(EXIT_FAILURE);
                break;
            default:
                printf("ERROR: invalid option\n");
                exit(EXIT_FAILURE);
        }
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

        cout  << "GPU id:" << n << "; name: " << dev_name 
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

    ASSERT_EQ(check_gdr_support(dev), true);

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    cout << "device ptr: 0x" << hex << d_A << dec << endl;
    cout << "allocated size: " << size << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    uint32_t *init_buf = NULL;
    uint32_t *h_buf = NULL;
    ASSERTDRV(cuMemAllocHost((void **)&init_buf, size));
    ASSERT_NEQ(init_buf, (void*)0);
    ASSERTDRV(cuMemAllocHost((void **)&h_buf, size));
    ASSERT_NEQ(h_buf, (void*)0);
    init_hbuf_walking_bit(init_buf, size);

    if (do_cumemcpy) {
        cout << endl;
        cout << "cuMemcpy_H2D num iters for each size: " << small_size_iter_factor * num_write_iters << "/" << num_write_iters << endl;
        printf("Test \t\t Size(B) \t Avg.Time(us) \t Avg.BW(MB/s)\n");
        BEGIN_CHECK {
            // cuMemcpy H2D benchmark
            copy_size = 1;
            while (copy_size <= size) {
                int iter = 0;
                size_t num_iters = (size < 100000 ? num_write_iters*small_size_iter_factor: num_write_iters);
                for (iter = 0; iter < num_iters+warmup; ++iter) {
                    if (iter == warmup)
                        clock_gettime(MYCLOCK, &beg);
                    ASSERTDRV(cuMemcpy(d_A, (CUdeviceptr)init_buf, copy_size));
                }
                clock_gettime(MYCLOCK, &end);
                double dt_us = (end.tv_nsec - beg.tv_nsec)/1000.0 + (end.tv_sec - beg.tv_sec)*1000000.0;
                lat_us =  dt_us / (double)num_iters;
                bw = copy_size / lat_us;
                printf("cuMemcpy_H2D \t %8zu \t %11.4f\t %11.4f\n", copy_size, lat_us, bw);
                copy_size <<= 1;
            }
        } END_CHECK;

        cout << endl;
        cout << "cuMemcpy_D2H num iters for each size: " << small_size_iter_factor * num_read_iters << "/" << num_read_iters << endl;
        printf("Test \t\t Size(B) \t Avg.Time(us) \t Avg.BW(MB/s)\n");
        BEGIN_CHECK {
            // cuMemcpy D2H benchmark
            copy_size = 1;
            while (copy_size <= size) {
                int iter = 0;
                size_t num_iters = (size < 100000 ? small_size_iter_factor*num_read_iters:num_read_iters);
                for (iter = 0; iter < num_iters+warmup; ++iter) {
                    if (iter == warmup)
                        clock_gettime(MYCLOCK, &beg);
                    ASSERTDRV(cuMemcpy((CUdeviceptr)h_buf, d_A, copy_size));
                }
                clock_gettime(MYCLOCK, &end);
                //lat_us = ((end.tv_nsec-beg.tv_nsec)/1000.0 + (end.tv_sec-beg.tv_sec)*1000000.0) / (double)iter;
                double dt_us = (end.tv_nsec - beg.tv_nsec)/1000.0 + (end.tv_sec - beg.tv_sec)*1000000.0;
                lat_us =  dt_us / (double)num_iters;
                bw = copy_size / lat_us;
                printf("cuMemcpy_D2H \t %8zu \t %11.4f\t %11.4f\n", copy_size, lat_us, bw);
                copy_size <<= 1;
            }
        } END_CHECK;

        cout << endl;
    }

    cout << endl;

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        ASSERT_EQ(gdr_pin_buffer(g, d_A, size, 0, 0, &mh), 0);
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
        cout << "user-space pointer: " << buf_ptr << endl;

        // gdr_copy_to_mapping benchmark
        cout << endl;
        cout << "gdr_copy_to_mapping num iters for each size: " << num_write_iters << endl;
        cout << "WARNING: Measuring the issue overhead as observed by the CPU. Data might not be ordered all the way to the GPU internal visibility." << endl;
        // For more information, see
        // https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#sync-behavior
        printf("Test \t\t\t Size(B) \t Avg.Time(us) \t Avg.BW(MB/s)\n");
        copy_size = 1;
        while (copy_size <= size) {
            int iter = 0;
            clock_gettime(MYCLOCK, &beg);
            size_t num_iters = (size < 100000 ? num_write_iters*small_size_iter_factor: num_write_iters);
            for (iter = 0; iter < num_iters+warmup; ++iter) {
                if (iter == warmup)
                    clock_gettime(MYCLOCK, &beg);
                gdr_copy_to_mapping(mh, buf_ptr, init_buf, copy_size);
            }
            clock_gettime(MYCLOCK, &end);
            double dt_us = (end.tv_nsec - beg.tv_nsec)/1000.0 + (end.tv_sec - beg.tv_sec)*1000000.0;
            lat_us =  dt_us / (double)num_iters;
            bw = copy_size / lat_us;
            printf("gdr_copy_to_mapping \t %8zu \t %11.4f\t %11.4f\n", copy_size, lat_us, bw);
            copy_size <<= 1;
        }

        MB();

        // gdr_copy_from_mapping benchmark
        cout << endl;
        cout << "gdr_copy_from_mapping num iters for each size: " << num_read_iters << endl;
        printf("Test \t\t\t Size(B) \t Avg.Time(us)\n");
        copy_size = 1;
        while (copy_size <= size) {
            int iter = 0;
            size_t num_iters = (size < 100000 ? small_size_iter_factor*num_read_iters:num_read_iters);
            for (iter = 0; iter < num_iters+warmup; ++iter) {
                if (iter == warmup)
                    clock_gettime(MYCLOCK, &beg);
                gdr_copy_from_mapping(mh, h_buf, buf_ptr, copy_size);
            }
            clock_gettime(MYCLOCK, &end);
            double dt_us = (end.tv_nsec - beg.tv_nsec)/1000.0 + (end.tv_sec - beg.tv_sec)*1000000.0;
            lat_us =  dt_us / (double)num_iters;
            bw = copy_size / lat_us;
            printf("gdr_copy_from_mapping \t %8zu \t %11.4f\t %11.4f\n", copy_size, lat_us, bw);
            copy_size <<= 1;
        }

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
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
