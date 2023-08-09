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

// Measuring the ping-pong latency of flag only.
__global__ void pp_kernel(uint32_t *gpu_flag_buf, uint32_t *cpu_flag_buf, uint32_t num_iters)
{
    uint32_t i = 1;
    WRITE_ONCE(*cpu_flag_buf, i);
    __threadfence_system();
    while (i < num_iters) {
        uint32_t val;
        do {
            val = READ_ONCE(*gpu_flag_buf);
        }
        while (val != i);

        ++val;
        WRITE_ONCE(*cpu_flag_buf, val);

        i = val;
    }
}

// This kernel emulates data + flag model. We consume the data by copying it to another GPU buffer.
__global__ void pp_data_kernel(uint32_t *gpu_flag_buf, uint32_t *cpu_flag_buf, uint32_t num_iters, uint32_t *A, uint32_t *B, size_t data_size)
{
    uint64_t my_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = gridDim.x * blockDim.x;
    uint64_t num_elements = data_size / sizeof(*A);
    uint32_t data_val;
    uint32_t flag_val;
    uint32_t i = 1;

    if (threadIdx.x == 0) {
        WRITE_ONCE(cpu_flag_buf[blockIdx.x], i);
        __threadfence_system();
    }
    __syncthreads();

    for (; i < num_iters; ++i) {
        if (threadIdx.x == 0) {
            do {
                flag_val = READ_ONCE(*gpu_flag_buf);
            }
            while (flag_val != i);
            __threadfence_system();
        }
        __syncthreads();

        for (uint64_t idx = my_tid; idx < num_elements; idx += num_threads)
            B[idx] = A[idx];
        __syncthreads();

        if (threadIdx.x == 0) {
            ++flag_val;
            WRITE_ONCE(cpu_flag_buf[blockIdx.x], flag_val);
            __threadfence_system();
        }
    }
}

static int dev_id = 0;
static uint32_t num_iters = 1000;
static size_t data_size = 0;

static unsigned int num_blocks = 8;
static unsigned int num_threads_per_block = 1024;

static unsigned int timeout = 10;  // in s
// Counter value before checking timeout.
static unsigned long int timeout_check_threshold = 1000000UL;
static unsigned long int timeout_counter = 0;

static void print_usage(const char *path)
{
    cout << "Usage: " << path << " [options]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h              Print this help text" << endl;
    cout << "   -d <gpu>        GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -t <iters>      Number of iterations (default: " << num_iters << ")" << endl;
    cout << "   -u <timeout>    Timeout in second. 0 to disable. (default: " << timeout << ")" << endl;
    cout << "   -a <fn>         GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                       Choices: cuMemAlloc, cuMemCreate" << endl;
    cout << "   -s <size>       Data size (default: " << data_size << ")" << endl;
    cout << "                       0 means measuring the visibility latency of the flag" << endl;
    cout << "   -B <nblocks>    Number of CUDA blocks (default: " << num_blocks << ")" << endl;
    cout << "   -T <nthreads>   Number of threads per CUDA blocks (default: " << num_threads_per_block << ")" << endl;
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
    uint32_t *g_gpu_flag_buf = NULL;
    uint32_t *h_cpu_flag_buf = NULL;

    CUdeviceptr d_gpu_flag_buf;
    CUdeviceptr d_cpu_flag_buf;

    gpu_mem_handle_t gpu_flag_mhandle;

    uint32_t *g_A;

    CUdeviceptr d_A = 0;
    CUdeviceptr d_B = 0;

    gpu_mem_handle_t A_mhandle;
    gpu_mem_handle_t B_mhandle;

    size_t data_buffer_size = 0;

    uint32_t *init_buf = NULL;

    struct timespec beg, end;
    double lat_us;
    double timeout_us;

    gpu_memalloc_fn_t galloc_fn = gpu_mem_alloc;
    gpu_memfree_fn_t gfree_fn = gpu_mem_free;

    while(1) {
        int c;
        c = getopt(argc, argv, "d:t:u:a:s:B:T:h");
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
            case 's':
                data_size = strtol(optarg, NULL, 0);
                break;
            case 'B':
                num_blocks = strtol(optarg, NULL, 0);
                break;
            case 'T':
                num_threads_per_block = strtol(optarg, NULL, 0);
                break;
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            default:
                cerr << "ERROR: invalid option" << endl;
                exit(EXIT_FAILURE);
        }
    }

    const bool do_consume_data = (data_size > 0);

    if (data_size % sizeof(*g_A) != 0) {
        cerr << "ERROR: data_size must be divisible by " << sizeof(*g_A) << "." << endl;
        exit(EXIT_FAILURE);
    }

    if (num_blocks <= 0) {
        cerr << "ERROR: nblocks must be at least 1." << endl;
        exit(EXIT_FAILURE);
    }

    if (num_threads_per_block <= 0) {
        cerr << "ERROR: nthreads must be at least 1." << endl;
        exit(EXIT_FAILURE);
    }

    timeout_us = timeout * 1000000.0;

    ASSERTDRV(cuInit(0));

    int n_devices = 0;
    ASSERTDRV(cuDeviceGetCount(&n_devices));

    CUdevice dev;
    for (int n = 0; n < n_devices; ++n) {

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

    int max_threads_per_block;
    ASSERTDRV(cuDeviceGetAttribute(&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));

    if (num_threads_per_block > max_threads_per_block) {
        cerr << "ERROR: nthreads can be at most " << max_threads_per_block << "." << endl;
        exit(EXIT_FAILURE);
    }

    if (!do_consume_data) {
        cout << "We will measure the visibility of the flag value only. "
             << "Setting nblocks and nthreads to 1."
             << endl;
        num_blocks = 1;
        num_threads_per_block = 1;
    }

    ASSERT_EQ(check_gdr_support(dev), true);

    if (galloc_fn == gpu_mem_alloc)
        cout << "gpu alloc fn: cuMemAlloc" << endl;
    else
        cout << "gpu alloc fn: cuMemCreate" << endl;

    if (do_consume_data) {
        data_buffer_size = PAGE_ROUND_UP(data_size, GPU_PAGE_SIZE);

        ASSERTDRV(galloc_fn(&A_mhandle, data_buffer_size, true, true));
        d_A = A_mhandle.ptr;
        cout << "d_A device ptr: 0x" << hex << d_A << dec << endl;

        ASSERTDRV(galloc_fn(&B_mhandle, data_buffer_size, true, true));
        d_B = B_mhandle.ptr;
        cout << "d_B device ptr: 0x" << hex << d_B << dec << endl;

        ASSERTDRV(cuMemAllocHost((void **)&init_buf, data_size));
        ASSERT_NEQ(init_buf, (void*)0);

        // Just set it to a random value. We don't use the content anyway.
        memset(init_buf, 0xaf, data_size);
    }

    ASSERTDRV(galloc_fn(&gpu_flag_mhandle, sizeof(*g_gpu_flag_buf), true, true));
    d_gpu_flag_buf = gpu_flag_mhandle.ptr;
    cout << "gpu flag device ptr: 0x" << hex << d_gpu_flag_buf << dec << endl;

    ASSERTDRV(cuMemsetD8(d_gpu_flag_buf, 0, sizeof(*g_gpu_flag_buf)));

    ASSERTDRV(cuMemHostAlloc((void **)&h_cpu_flag_buf, sizeof(*h_cpu_flag_buf) * num_blocks, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
    ASSERT_NEQ(h_cpu_flag_buf, (void*)0);
    ASSERTDRV(cuMemHostGetDevicePointer(&d_cpu_flag_buf, h_cpu_flag_buf, 0));
    memset(h_cpu_flag_buf, 0, sizeof(*h_cpu_flag_buf) * num_blocks);


    gdr_t g = gdr_open_safe();

    gdr_mh_t gpu_flag_mh;
    void *map_gpu_flag_ptr = NULL;
    gdr_info_t gpu_flag_info;
    int gpu_flag_off;

    gdr_mh_t A_mh;
    void *map_A_ptr = NULL;
    gdr_info_t A_info;
    int A_off;

    BEGIN_CHECK {
        // tokens are optional in CUDA 6.0
        ASSERT_EQ(gdr_pin_buffer(g, d_gpu_flag_buf, sizeof(*g_gpu_flag_buf), 0, 0, &gpu_flag_mh), 0);
        ASSERT_NEQ(gpu_flag_mh, null_mh);

        ASSERT_EQ(gdr_map(g, gpu_flag_mh, &map_gpu_flag_ptr, sizeof(*g_gpu_flag_buf)), 0);
        cout << "map_gpu_flag_ptr: " << map_gpu_flag_ptr << endl;

        ASSERT_EQ(gdr_get_info(g, gpu_flag_mh, &gpu_flag_info), 0);
        cout << "gpu_flag_info.va: " << hex << gpu_flag_info.va << dec << endl;
        cout << "gpu_flag_info.mapped_size: " << gpu_flag_info.mapped_size << endl;
        cout << "gpu_flag_info.page_size: " << gpu_flag_info.page_size << endl;
        cout << "gpu_flag_info.mapped: " << gpu_flag_info.mapped << endl;
        cout << "gpu_flag_info.wc_mapping: " << gpu_flag_info.wc_mapping << endl;

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        gpu_flag_off = gpu_flag_info.va - d_gpu_flag_buf;
        cout << "gpu_flag page offset: " << gpu_flag_off << endl;

        g_gpu_flag_buf = (uint32_t *)((uintptr_t)map_gpu_flag_ptr + gpu_flag_off);
        cout << "gpu_flag user-space pointer: " << g_gpu_flag_buf << endl;

        if (do_consume_data) {
            ASSERT_EQ(gdr_pin_buffer(g, d_A, data_buffer_size, 0, 0, &A_mh), 0);
            ASSERT_NEQ(A_mh, null_mh);

            ASSERT_EQ(gdr_map(g, A_mh, &map_A_ptr, data_buffer_size), 0);
            cout << "map_A_ptr: " << map_A_ptr << endl;

            ASSERT_EQ(gdr_get_info(g, A_mh, &A_info), 0);
            cout << "A_info.va: " << hex << A_info.va << dec << endl;
            cout << "A_info.mapped_size: " << A_info.mapped_size << endl;
            cout << "A_info.page_size: " << A_info.page_size << endl;
            cout << "A_info.mapped: " << A_info.mapped << endl;
            cout << "A_info.wc_mapping: " << A_info.wc_mapping << endl;

            // remember that mappings start on a 64KB boundary, so let's
            // calculate the offset from the head of the mapping to the
            // beginning of the buffer
            A_off = A_info.va - d_A;
            cout << "A page offset: " << A_off << endl;

            g_A = (uint32_t *)((uintptr_t)map_A_ptr + A_off);
            cout << "A user-space pointer: " << (void *)g_A << endl;

            cout << "Measuring the latency of data + flag model." << endl
                 << "CPU does gdr_copy_to_mapping for data followed by gdr_copy_to_mapping for flag. "
                 << "GPU polls on the flag. Upon observing the upate, it consumes the data. "
                 << "When everything is done, GPU notifies CPU back via cuMemHostAlloc'd buffer." << endl
                 << "We report the round-trip time from when CPU starts writing the data until it observes the notification from GPU." << endl
                 << endl;

            cout << "Running " << num_iters << " iterations with data size "
                 << data_size << " bytes and flag size " << sizeof(*g_gpu_flag_buf) << " bytes."
                 << endl;

            pp_data_kernel<<< num_blocks, num_threads_per_block >>>((uint32_t *)d_gpu_flag_buf, (uint32_t *)d_cpu_flag_buf, num_iters, (uint32_t *)d_A, (uint32_t *)d_B, data_size);
        }
        else {
            cout << "Measuring the visibility latency of the flag value." << endl
                 << "CPU does gdr_copy_to_mapping, and GPU notifies back via cuMemHostAlloc'd buffer." << endl
                 << "We report the round-trip time from when CPU updates the flag value until it observes the notification from GPU." << endl
                 << endl;

            cout << "Running " << num_iters << " iterations with flag size " << sizeof(*g_gpu_flag_buf) << " bytes." << endl;

            pp_kernel<<< num_blocks, num_threads_per_block >>>((uint32_t *)d_gpu_flag_buf, (uint32_t *)d_cpu_flag_buf, num_iters);
        }

        // Catching any potential errors. CUDA_ERROR_NOT_READY means the kernel
        // is running. We expect to see this status instead of CUDA_SUCCESS
        // because the kernel must wait for signal from CPU, which occurs after
        // this line.
        ASSERT_EQ(cuStreamQuery(0), CUDA_ERROR_NOT_READY);

        uint32_t i = 1;
        uint32_t val;
        unsigned int cpu_flag_idx = 0;
        // Wait for pp_kernel to be ready before starting the time measurement.
        clock_gettime(MYCLOCK, &beg);
        do {
            val = READ_ONCE(h_cpu_flag_buf[cpu_flag_idx]);
            if (val == i)
                ++cpu_flag_idx;
            else
                check_timeout(beg, timeout_us);
        }
        while (cpu_flag_idx < num_blocks);
        LB();

        // Restart the timer for measurement.
        clock_gettime(MYCLOCK, &beg);
        while (i < num_iters) {
            if (do_consume_data) {
                gdr_copy_to_mapping(A_mh, g_A, init_buf, data_size);
                SB();
            }
            gdr_copy_to_mapping(gpu_flag_mh, g_gpu_flag_buf, &val, sizeof(g_gpu_flag_buf));
            SB();

            cpu_flag_idx = 0;
            do {
                val = READ_ONCE(h_cpu_flag_buf[cpu_flag_idx]);
                if (val == i + 1)
                    ++cpu_flag_idx;
                else
                    check_timeout(beg, timeout_us);
            }
            while (cpu_flag_idx < num_blocks);
            LB();
            i = val;
        }
        clock_gettime(MYCLOCK, &end);

        ASSERTDRV(cuStreamSynchronize(0));

        clock_gettime(MYCLOCK, &end);
        lat_us = time_diff(beg, end) / (double)num_iters;

        cout << "Round-trip latency per iteration is " << lat_us << " us" << endl;

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, gpu_flag_mh, map_gpu_flag_ptr, sizeof(*g_gpu_flag_buf)), 0);
        if (do_consume_data)
            ASSERT_EQ(gdr_unmap(g, A_mh, map_A_ptr, data_buffer_size), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, gpu_flag_mh), 0);
        if (do_consume_data)
            ASSERT_EQ(gdr_unpin_buffer(g, A_mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFreeHost(h_cpu_flag_buf));
    ASSERTDRV(gfree_fn(&gpu_flag_mhandle));

    if (do_consume_data) {
        ASSERTDRV(gfree_fn(&A_mhandle));
        ASSERTDRV(gfree_fn(&B_mhandle));
        ASSERTDRV(cuMemFreeHost(init_buf));
    }

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
