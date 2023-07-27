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

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

using namespace gdrcopy::test;

// manually tuned...
int num_iters        = 100;
int num_bins         = 10;
int num_warmup_iters = 10;
size_t _size = (size_t)1 << 24;
int dev_id = 0;

void print_usage(const char *path)
{
    cout << "Usage: " << path << " [-h][-s <max-size>][-d <gpu>][-n <iters>][-w <iters>][-a <fn>]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h              Print this help text" << endl;
    cout << "   -s <max-size>   Max buffer size to benchmark (default: " << _size << ")" << endl;
    cout << "   -d <gpu>        GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -n <iters>      Number of benchmark iterations (default: " << num_iters << ")" << endl;
    cout << "   -w <iters>      Number of warm-up iterations (default: " << num_warmup_iters << ")" << endl;
    cout << "   -a <fn>         GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                       Choices: cuMemAlloc, cuMemCreate" << endl;
}

void run_test(CUdeviceptr d_A, size_t size)
{
    // minimum pinning size is a GPU page size
    size_t pin_request_size = GPU_PAGE_SIZE;
    struct timespec beg, end;
    double pin_lat_us;
    double map_lat_us;
    double unpin_lat_us;
    double unmap_lat_us;
    double inf_lat_us;
    double delta_lat_us;
    double *lat_arr;
    int *bin_arr;

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled

        lat_arr = (double *)malloc(sizeof(double) * num_iters);
        bin_arr = (int *)malloc(sizeof(double) * num_bins);

        while (pin_request_size <= size) {
            int iter = 0;
            size_t actual_pin_size;
            double min_lat, max_lat;
            min_lat = -1;
            max_lat = -1;
            pin_lat_us = 0;
            map_lat_us = 0;
            unpin_lat_us = 0;
            unmap_lat_us = 0;
            inf_lat_us = 0;
            actual_pin_size = PAGE_ROUND_UP(pin_request_size, GPU_PAGE_SIZE);

            for (iter = 0; iter < num_warmup_iters; ++iter) {

                BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, actual_pin_size, 0, 0, &mh), 0);
                ASSERT_NEQ(mh, null_mh);

                void *map_d_ptr  = NULL;
                ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, actual_pin_size), 0);

                gdr_info_t info;
                ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
                ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, actual_pin_size), 0);
                ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
            }

            for (iter = 0; iter < num_iters; ++iter) {

                clock_gettime(MYCLOCK, &beg);
                BREAK_IF_NEQ(gdr_pin_buffer(g, d_A, actual_pin_size, 0, 0, &mh), 0);
                clock_gettime(MYCLOCK, &end);
                delta_lat_us = ((end.tv_nsec-beg.tv_nsec)/1000.0 + (end.tv_sec-beg.tv_sec)*1000000.0);
                pin_lat_us += delta_lat_us;
                ASSERT_NEQ(mh, null_mh);
                lat_arr[iter] = delta_lat_us;
                min_lat = (min_lat == -1) ? delta_lat_us : ((delta_lat_us < min_lat) ? delta_lat_us : min_lat);
                max_lat = delta_lat_us > max_lat ? delta_lat_us : max_lat;

                void *map_d_ptr  = NULL;
                clock_gettime(MYCLOCK, &beg);
                ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, actual_pin_size), 0);
                clock_gettime(MYCLOCK, &end);
                delta_lat_us = ((end.tv_nsec-beg.tv_nsec)/1000.0 + (end.tv_sec-beg.tv_sec)*1000000.0);
                map_lat_us += delta_lat_us;

                gdr_info_t info;
                clock_gettime(MYCLOCK, &beg);
                ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
                clock_gettime(MYCLOCK, &end);
                delta_lat_us = ((end.tv_nsec-beg.tv_nsec)/1000.0 + (end.tv_sec-beg.tv_sec)*1000000.0);
                inf_lat_us += delta_lat_us;

                clock_gettime(MYCLOCK, &beg);
                ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, actual_pin_size), 0);
                clock_gettime(MYCLOCK, &end);
                delta_lat_us = ((end.tv_nsec-beg.tv_nsec)/1000.0 + (end.tv_sec-beg.tv_sec)*1000000.0);
                unmap_lat_us += delta_lat_us;

                clock_gettime(MYCLOCK, &beg);
                ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
                clock_gettime(MYCLOCK, &end);
                delta_lat_us = ((end.tv_nsec-beg.tv_nsec)/1000.0 + (end.tv_sec-beg.tv_sec)*1000000.0);
                unpin_lat_us += delta_lat_us;
            }

            pin_lat_us /= iter;
            map_lat_us /= iter;
            inf_lat_us /= iter;
            unpin_lat_us /= iter;
            unmap_lat_us /= iter;

            printf("Size(B)\tpin.Time(us)\tmap.Time(us)\tget_info.Time(us)\tunmap.Time(us)\tunpin.Time(us)\n");
            printf("%zu\t%f\t%f\t%f\t%f\t%f\n",
                    actual_pin_size, pin_lat_us, map_lat_us, inf_lat_us, unmap_lat_us, unpin_lat_us);
            pin_request_size <<= 1;

            printf("Histogram of gdr_pin_buffer latency for %ld bytes\n", actual_pin_size);
            print_histogram(lat_arr, num_iters, bin_arr, num_bins, min_lat, max_lat);
            printf("\n");
        }

        free(lat_arr);
        free(bin_arr);
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
        c = getopt(argc, argv, "s:d:n:w:a:h");
        if (c == -1)
            break;

        switch (c) {
            case 's':
                _size = strtol(optarg, NULL, 0);
                break;
            case 'd':
                dev_id = strtol(optarg, NULL, 0);
                break;
            case 'n':
                num_iters = strtol(optarg, NULL, 0);
                break;
            case 'w':
                num_warmup_iters = strtol(optarg, NULL, 0);
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
                break;
            default:
                printf("ERROR: invalid option\n");
                exit(EXIT_FAILURE);
        }
    }

    size_t size = PAGE_ROUND_UP(_size, GPU_PAGE_SIZE);

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
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;
    cout << "device ptr: 0x" << hex << d_A << dec << endl;
    cout << "allocated size: " << size << endl;

    run_test(d_A, size);

    ASSERTDRV(gfree_fn(&mhandle));

    ASSERTDRV(cuCtxSetCurrent(NULL));
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
