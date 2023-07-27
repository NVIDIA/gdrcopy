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

__global__ void pp_kernel(uint32_t *d_buf, uint32_t *h_buf, uint32_t num_iters)
{
    uint32_t i = 1;
    WRITE_ONCE(*h_buf, i);
    __threadfence_block();
    while (i < num_iters) {
        while (READ_ONCE(*d_buf) != i) ;
        __threadfence_block();

        ++i;

        WRITE_ONCE(*h_buf, i);
        __threadfence_block();
    }
}

static int dev_id = 0;
static uint32_t num_iters = 1000;
static unsigned int timeout = 10;  // in s
// Counter value before checking timeout.
static unsigned long int timeout_check_threshold = 1000000UL;
static unsigned long int timeout_counter = 0;

static void print_usage(const char *path)
{
    cout << "Usage: " << path << " [-h][-d <gpu>][-t <iters>][-u <timeout>][-a <fn>]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h              Print this help text" << endl;
    cout << "   -d <gpu>        GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -t <iters>      Number of iterations (default: " << num_iters << ")" << endl;
    cout << "   -u <timeout>    Timeout in second. 0 to disable. (default: " << timeout << ")" << endl;
    cout << "   -a <fn>         GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                       Choices: cuMemAlloc, cuMemCreate" << endl;
}

/**
 * Return time difference in us.
 */
static inline double time_diff(struct timespec start, struct timespec end)
{
    return (double)((end.tv_nsec - start.tv_nsec) / 1000.0 + (end.tv_sec - start.tv_sec) * 1000000.0);
}

static inline void check_timeout(struct timespec start, double timeout_us)
{
    CUresult status;
    const char *cu_status_name;
    struct timespec now;
    double time_used_us;
    if (timeout_us > 0) {
        ++timeout_counter;
        if (timeout_counter >= timeout_check_threshold) {
            clock_gettime(MYCLOCK, &now);
            time_used_us = time_diff(start, now);
            if (time_used_us > timeout_us) {
                cerr << "ERROR: TIMEOUT!!!" << endl;
                status = cuStreamQuery(0);
                cuGetErrorName(status, &cu_status_name);
                cerr << "cuStreamQuery(0) returned " << cu_status_name << endl;
                abort();
            }
            timeout_counter = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    uint32_t *d_buf = NULL;
    uint32_t *h_buf = NULL;

    CUdeviceptr d_buf_cuptr;
    CUdeviceptr h_buf_cuptr;

    gpu_mem_handle_t mhandle;

    struct timespec beg, end;
    double lat_us;
    double timeout_us;

    gpu_memalloc_fn_t galloc_fn = gpu_mem_alloc;
    gpu_memfree_fn_t gfree_fn = gpu_mem_free;

    while(1) {        
        int c;
        c = getopt(argc, argv, "d:t:u:a:h");
        if (c == -1)
            break;

        switch (c) {
            case 'd':
                dev_id = strtol(optarg, NULL, 0);
                break;
            case 't':
                num_iters = strtol(optarg, NULL, 0);
                break;
            case 'u':
                timeout = strtol(optarg, NULL, 0);
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
                printf("ERROR: invalid option\n");
                exit(EXIT_FAILURE);
        }
    }

    timeout_us = timeout * 1000000.0;

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

    ASSERTDRV(galloc_fn(&mhandle, sizeof(*d_buf), true, true));
    d_buf_cuptr = mhandle.ptr;
    cout << "device ptr: 0x" << hex << d_buf_cuptr << dec << endl;

    if (galloc_fn == gpu_mem_alloc)
        cout << "gpu alloc fn: cuMemAlloc" << endl;
    else
        cout << "gpu alloc fn: cuMemCreate" << endl;

    ASSERTDRV(cuMemsetD8(d_buf_cuptr, 0, sizeof(*d_buf)));

    ASSERTDRV(cuMemHostAlloc((void **)&h_buf, sizeof(*h_buf), CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    ASSERT_NEQ(h_buf, (void*)0);
    ASSERTDRV(cuMemHostGetDevicePointer(&h_buf_cuptr, h_buf, 0));
    memset(h_buf, 0, sizeof(*h_buf));


    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        ASSERT_EQ(gdr_pin_buffer(g, d_buf_cuptr, sizeof(*d_buf), 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &map_d_ptr, sizeof(*d_buf)), 0);
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
        int off = info.va - d_buf_cuptr;
        cout << "page offset: " << off << endl;

        d_buf = (uint32_t *)((uintptr_t)map_d_ptr + off);
        cout << "user-space pointer: " << d_buf << endl;

        cout << "CPU does gdr_copy_to_mapping and GPU writes back via cuMemHostAlloc'd buffer." << endl;
        cout << "Running " << num_iters << " iterations with data size " << sizeof(*d_buf) << " bytes." << endl;

        pp_kernel<<< 1, 1 >>>((uint32_t *)d_buf_cuptr, (uint32_t *)h_buf_cuptr, num_iters);

        // Catching any potential errors. CUDA_ERROR_NOT_READY means pp_kernel
        // is running. We expect to see this status instead of CUDA_SUCCESS
        // because pp_kernel must wait for signal from CPU, which occurs after
        // this line.
        ASSERT_EQ(cuStreamQuery(0), CUDA_ERROR_NOT_READY);

        uint32_t i = 1;
        // Wait for pp_kernel to be ready before starting the time measurement.
        clock_gettime(MYCLOCK, &beg);
        while (READ_ONCE(*h_buf) != i) {
            check_timeout(beg, timeout_us);
        }
        LB();

        // Restart the timer for measurement.
        clock_gettime(MYCLOCK, &beg);
        while (i < num_iters) {
            gdr_copy_to_mapping(mh, d_buf, &i, sizeof(d_buf));
            SB();

            ++i;

            while (READ_ONCE(*h_buf) != i) {
                check_timeout(beg, timeout_us);
            }
            LB();
        }
        clock_gettime(MYCLOCK, &end);

        ASSERTDRV(cuStreamSynchronize(0));

        clock_gettime(MYCLOCK, &end);
        lat_us = time_diff(beg, end) / (double)num_iters;

        cout << "Round-trip latency per iteration is " << lat_us << " us" << endl;

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, sizeof(*d_buf)), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFreeHost(h_buf));
    ASSERTDRV(gfree_fn(&mhandle));

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
