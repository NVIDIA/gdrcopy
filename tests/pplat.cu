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

// In nanoseconds
__device__ static inline uint64_t query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
    return ret;
}

// Measuring the ping-pong latency of flag only.
__global__ void pp_cpu_produce_gpu_consume_kernel(uint32_t *gpu_flag_buf, uint32_t *cpu_flag_buf, uint32_t num_iters)
{
    uint32_t i = 1;
    WRITE_ONCE(*cpu_flag_buf, i);
    __threadfence_system();
    while (i <= num_iters) {
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

__global__ void pp_gpu_produce_cpu_consume_kernel(uint32_t *gpu_flag_buf, uint32_t *cpu_flag_buf, uint32_t num_iters, uint64_t *beg, uint64_t *end)
{
    uint32_t val;
    uint32_t i = 1;
    uint64_t _beg, _end;
    WRITE_ONCE(*cpu_flag_buf, i);
    __threadfence_system();

    do {
        val = READ_ONCE(*gpu_flag_buf);
    }
    while (val != i);
    __threadfence_system();

    i = val + 1;
    _beg = query_globaltimer();
    while (i <= num_iters + 1) {
        val = i;
        WRITE_ONCE(*cpu_flag_buf, val);

        do {
            val = READ_ONCE(*gpu_flag_buf);
        }
        while (val != i);

        ++val;

        i = val;
    }
    _end = query_globaltimer();

    *beg = _beg;
    *end = _end;
}

// This kernel emulates data + flag model. We consume the data by summing all
// elements to the cpu_flag. The values of all data elements are zero. So, it
// does not affect the outcome. This is just for creating data dependency.
__global__ void pp_data_kernel(uint32_t *gpu_flag_buf, uint32_t *cpu_flag_buf, uint32_t num_iters, uint32_t *A, size_t data_size)
{
    uint64_t my_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = gridDim.x * blockDim.x;
    uint64_t num_elements = data_size / sizeof(*A);
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
            flag_val += READ_ONCE(A[idx]);
        __syncthreads();

        if (threadIdx.x == 0) {
            ++flag_val;
            WRITE_ONCE(cpu_flag_buf[blockIdx.x], flag_val);
            __threadfence_system();
        }
    }
}

typedef enum {
    MEM_LOC_GPU = 0,
    MEM_LOC_HOST
} mem_loc_t;

typedef enum {
    BENCHMARK_MODE_CPU_PRODUCE_GPU_CONSUME = 0,
    BENCHMARK_MODE_GPU_PRODUCE_CPU_CONSUME,
} benchmark_mode_t;

typedef struct {
    CUdeviceptr gpu_ptr;
    void *host_ptr;
    size_t size;
    mem_loc_t mem_loc;

    // GDRCopy-related objects
    gpu_mem_handle_t gmhandle;
    gdr_t g;
    gdr_mh_t mh;
    void *map_ptr;
} gh_mem_handle_t;

static int dev_id = 0;
static uint32_t num_iters = 1000;
static size_t data_size = 0;

static unsigned int num_blocks = 8;
static unsigned int num_threads_per_block = 1024;

static mem_loc_t gpu_flag_loc = MEM_LOC_GPU;
static mem_loc_t cpu_flag_loc = MEM_LOC_HOST;
static mem_loc_t data_buf_loc = MEM_LOC_GPU;

static gpu_memalloc_fn_t galloc_fn = gpu_mem_alloc;
static gpu_memfree_fn_t gfree_fn = gpu_mem_free;

static benchmark_mode_t benchmark_mode = BENCHMARK_MODE_CPU_PRODUCE_GPU_CONSUME;

static unsigned int timeout = 10;  // in s
// Counter value before checking timeout.
static unsigned long int timeout_check_threshold = 1000000UL;
static unsigned long int timeout_counter = 0;

static inline string mem_loc_to_str(mem_loc_t loc)
{
    switch (loc) {
        case MEM_LOC_GPU:
            return string("gpumem");
        case MEM_LOC_HOST:
            return string("hostmem");
        default:
            cerr << "Unrecognized loc" << endl;
            exit(EXIT_FAILURE);
    }
}

static inline int str_to_mem_loc(mem_loc_t *loc, string s)
{
    int status = 0;
    if (s == "gpumem")
        *loc = MEM_LOC_GPU;
    else if (s == "hostmem")
        *loc = MEM_LOC_HOST;
    else
        status = EINVAL;
    return status;
}

static void gh_mem_alloc(gh_mem_handle_t *mhandle, size_t size, mem_loc_t loc, gdr_t g)
{
    gpu_mem_handle_t gmhandle;
    CUdeviceptr gpu_ptr;
    void *host_ptr = NULL;
    gdr_mh_t mh;
    void *map_ptr = NULL;

    if (loc == MEM_LOC_GPU) {
        gdr_info_t info;
        off_t off;

        ASSERTDRV(galloc_fn(&gmhandle, size, true, true));
        gpu_ptr = gmhandle.ptr;

        ASSERTDRV(cuMemsetD8(gpu_ptr, 0, size));

        // tokens are optional in CUDA 6.0
        ASSERT_EQ(gdr_pin_buffer(g, gpu_ptr, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        ASSERT_EQ(gdr_map(g, mh, &map_ptr, size), 0);

        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        off = info.va - gpu_ptr;

        host_ptr = (void *)((uintptr_t)map_ptr + off);
    }
    else if (loc == MEM_LOC_HOST) {
        ASSERTDRV(cuMemHostAlloc(&host_ptr, size, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
        ASSERT_NEQ(host_ptr, (void*)0);
        ASSERTDRV(cuMemHostGetDevicePointer(&gpu_ptr, host_ptr, 0));
        memset(host_ptr, 0, size);
    }
    else {
        cerr << "Unrecognized loc" << endl;
        exit(EXIT_FAILURE);
    }

    mhandle->gpu_ptr = gpu_ptr;
    mhandle->host_ptr = host_ptr;
    mhandle->size = size;
    mhandle->mem_loc = loc;
    mhandle->gmhandle = gmhandle;
    mhandle->g = g;
    mhandle->mh = mh;
    mhandle->map_ptr = map_ptr;
}

static void gh_mem_free(gh_mem_handle_t *mhandle)
{
    if (mhandle->mem_loc == MEM_LOC_GPU) {
        ASSERT_EQ(gdr_unmap(mhandle->g, mhandle->mh, mhandle->map_ptr, mhandle->size), 0);
        ASSERT_EQ(gdr_unpin_buffer(mhandle->g, mhandle->mh), 0);
        ASSERTDRV(gfree_fn(&mhandle->gmhandle));
    }
    else if (mhandle->mem_loc == MEM_LOC_HOST) {
        ASSERTDRV(cuMemFreeHost(mhandle->host_ptr));
    }
    else {
        cerr << "Unrecognized loc" << endl;
        exit(EXIT_FAILURE);
    }
}

static void print_usage(const char *path)
{
    cout << "Usage: " << path << " [options]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h                  Print this help text" << endl;
    cout << "   -d <gpu>            GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -t <iters>          Number of iterations (default: " << num_iters << ")" << endl;
    cout << "   -u <timeout>        Timeout in second. 0 to disable. (default: " << timeout << ")" << endl;
    cout << "   -a <fn>             GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                           Choices: cuMemAlloc, cuMemCreate" << endl;
    cout << "   -s <size>           Data size (default: " << data_size << ")" << endl;
    cout << "                           0 means measuring the visibility latency of the flag" << endl;
    cout << "   -B <nblocks>        Number of CUDA blocks (default: " << num_blocks << ")" << endl;
    cout << "   -T <nthreads>       Number of threads per CUDA blocks (default: " << num_threads_per_block << ")" << endl;
    cout << "   -m <mode>           Benchmark mode (default: " << benchmark_mode << ")" << endl;
    cout << "                           0: CPU produces and notifies GPU. GPU consumes and notifies CPU." << endl;
    cout << "                           1: GPU produces and notifies CPU. CPU consumes and notifies GPU." << endl;
    cout << "   -G <gpu-flag-loc>   The location of GPU flag (default: " << mem_loc_to_str(gpu_flag_loc) << ")" << endl;
    cout << "                           Choices: gpumem, hostmem" << endl;
    cout << "                           This flag is used by CPU to notify GPU." << endl;
    cout << "   -C <cpu-flag-loc>   The location of CPU flag (default: " << mem_loc_to_str(cpu_flag_loc) << ")" << endl;
    cout << "                           Choices: gpumem, hostmem" << endl;
    cout << "                           This flag is used by GPU to notify CPU." << endl;
    cout << "   -D <data-buf-loc>   The location of data buffer (default: " << mem_loc_to_str(data_buf_loc) << ")" << endl;
    cout << "                           Choices: gpumem, hostmem" << endl;
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
    struct timespec beg, end;
    double lat_us;
    double timeout_us;

    while (1) {
        int c;
        c = getopt(argc, argv, "d:t:u:a:s:B:T:m:G:C:D:h");
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
            case 'a': {
                string optarg_str = string(optarg);
                if (optarg_str == "cuMemAlloc") {
                    galloc_fn = gpu_mem_alloc;
                    gfree_fn = gpu_mem_free;
                }
                else if (optarg_str == "cuMemCreate") {
                    galloc_fn = gpu_vmm_alloc;
                    gfree_fn = gpu_vmm_free;
                }
                else {
                    cerr << "Unrecognized fn argument" << endl;
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 's':
                data_size = strtol(optarg, NULL, 0);
                break;
            case 'B':
                num_blocks = strtol(optarg, NULL, 0);
                break;
            case 'T':
                num_threads_per_block = strtol(optarg, NULL, 0);
                break;
            case 'm':
                benchmark_mode = (benchmark_mode_t)strtol(optarg, NULL, 0);
                break;
            case 'G': {
                string optarg_str = string(optarg);
                int status = str_to_mem_loc(&gpu_flag_loc, optarg_str);
                if (status) {
                    cerr << "Unrecognized gpu-flag-loc argument" << endl;
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'C': {
                string optarg_str = string(optarg);
                int status = str_to_mem_loc(&cpu_flag_loc, optarg_str);
                if (status) {
                    cerr << "Unrecognized cpu-flag-loc argument" << endl;
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'D': {
                string optarg_str = string(optarg);
                int status = str_to_mem_loc(&data_buf_loc, optarg_str);
                if (status) {
                    cerr << "Unrecognized data-buf-loc argument" << endl;
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            default:
                cerr << "ERROR: invalid option" << endl;
                exit(EXIT_FAILURE);
        }
    }

    const bool process_data = (data_size > 0);

    if (data_size % sizeof(uint32_t) != 0) {
        cerr << "ERROR: data_size must be divisible by " << sizeof(uint32_t) << "." << endl;
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

    if (BENCHMARK_MODE_CPU_PRODUCE_GPU_CONSUME > benchmark_mode || BENCHMARK_MODE_GPU_PRODUCE_CPU_CONSUME < benchmark_mode) {
        cerr << "ERROR: Unrecognized mode " << benchmark_mode << "." << endl;
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

    if (!process_data) {
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

    gdr_t g = gdr_open_safe();

    gh_mem_handle_t gpu_flag_mhandle;
    gh_mem_handle_t cpu_flag_mhandle;
    gh_mem_handle_t data_buf_mhandle;
    gh_mem_handle_t init_buf_mhandle;
    gh_mem_handle_t gpu_beg_mhandle;
    gh_mem_handle_t gpu_end_mhandle;

    BEGIN_CHECK {
        gh_mem_alloc(&gpu_flag_mhandle, sizeof(uint32_t), gpu_flag_loc, g);
        gh_mem_alloc(&cpu_flag_mhandle, sizeof(uint32_t) * num_blocks, cpu_flag_loc, g);
        gh_mem_alloc(&gpu_beg_mhandle, sizeof(uint64_t) * num_blocks, MEM_LOC_GPU, g);
        gh_mem_alloc(&gpu_end_mhandle, sizeof(uint64_t) * num_blocks, MEM_LOC_GPU, g);

        if (process_data) {
            gh_mem_alloc(&data_buf_mhandle, data_size, data_buf_loc, g);
            gh_mem_alloc(&init_buf_mhandle, data_size, MEM_LOC_HOST, g);

            cout << "Measuring the latency of data + flag model." << endl
                 << "CPU does gdr_copy_to_mapping for data followed by gdr_copy_to_mapping for flag. "
                 << "GPU polls on the flag. Upon observing the upate, it consumes the data. "
                 << "When everything is done, GPU notifies CPU back via cuMemHostAlloc'd buffer." << endl
                 << "We report the round-trip time from when CPU starts writing the data until it observes the notification from GPU." << endl
                 << endl;

            cout << "Running " << num_iters << " iterations with data size "
                 << data_size << " bytes and flag size " << gpu_flag_mhandle.size << " bytes."
                 << endl;

            pp_data_kernel<<< num_blocks, num_threads_per_block >>>((uint32_t *)gpu_flag_mhandle.gpu_ptr, (uint32_t *)cpu_flag_mhandle.gpu_ptr, num_iters, (uint32_t *)data_buf_mhandle.gpu_ptr, data_size);
        }
        else {
            cout << "Measuring the visibility latency of the flag value." << endl
                 << "Running " << num_iters << " iterations with flag size " << gpu_flag_mhandle.size << " bytes." << endl
                 << endl;

            switch (benchmark_mode) {
                case BENCHMARK_MODE_CPU_PRODUCE_GPU_CONSUME:
                    cout << "CPU writes to gpu_flag. GPU polls on the expected gpu_flag value. "
                         << "GPU writes back via cpu_flag. CPU polls on the expected cpu_flag value. "
                         << "We report the round-trip time from when CPU writes to gpu_flag "
                         << "until it observes the update in cpu_flag." << endl
                         << "CPU does the time measurement." << endl
                         << endl;
                    pp_cpu_produce_gpu_consume_kernel<<< num_blocks, num_threads_per_block >>>((uint32_t *)gpu_flag_mhandle.gpu_ptr, (uint32_t *)cpu_flag_mhandle.gpu_ptr, num_iters);
                    break;
                case BENCHMARK_MODE_GPU_PRODUCE_CPU_CONSUME:
                    cout << "GPU writes to cpu_flag. CPU polls on the expected cpu_flag value. "
                         << "CPU writes back via gpu_flag. GPU polls on the expected gpu_flag value. "
                         << "We report the round-trip time from when GPU writes to cpu_flag "
                         << "until it observes the update in gpu_flag." << endl
                         << "GPU does the time measurement." << endl
                         << endl;
                    pp_gpu_produce_cpu_consume_kernel<<< num_blocks, num_threads_per_block >>>((uint32_t *)gpu_flag_mhandle.gpu_ptr, (uint32_t *)cpu_flag_mhandle.gpu_ptr, num_iters, (uint64_t *)gpu_beg_mhandle.gpu_ptr, (uint64_t *)gpu_end_mhandle.gpu_ptr);
                    break;
            }

        }

        // Catching any potential errors. CUDA_ERROR_NOT_READY means the kernel
        // is running. We expect to see this status instead of CUDA_SUCCESS
        // because the kernel must wait for signal from CPU, which occurs after
        // this line.
        ASSERT_EQ(cuStreamQuery(0), CUDA_ERROR_NOT_READY);

        uint32_t i = 1;
        uint32_t val;
        unsigned int cpu_flag_idx = 0;
        // Wait for the kernel to be ready before starting the time measurement.
        clock_gettime(MYCLOCK, &beg);
        do {
            if (cpu_flag_mhandle.mem_loc == MEM_LOC_GPU)
                gdr_copy_from_mapping(cpu_flag_mhandle.mh, &val, &((uint32_t *)cpu_flag_mhandle.host_ptr)[cpu_flag_idx], sizeof(uint32_t));
            else
                val = READ_ONCE(((uint32_t *)cpu_flag_mhandle.host_ptr)[cpu_flag_idx]);
            if (val == i)
                ++cpu_flag_idx;
            else
                check_timeout(beg, timeout_us);
        }
        while (cpu_flag_idx < num_blocks);
        LB();

        switch (benchmark_mode) {
            case BENCHMARK_MODE_CPU_PRODUCE_GPU_CONSUME: {
                // Restart the timer for measurement.
                clock_gettime(MYCLOCK, &beg);
                while (i <= num_iters) {
                    if (process_data) {
                        gdr_copy_to_mapping(data_buf_mhandle.mh, data_buf_mhandle.host_ptr, init_buf_mhandle.host_ptr, data_size);
                        SB();
                    }
                    if (gpu_flag_loc == MEM_LOC_GPU)
                        gdr_copy_to_mapping(gpu_flag_mhandle.mh, gpu_flag_mhandle.host_ptr, &val, sizeof(gpu_flag_mhandle.size));
                    else
                        WRITE_ONCE(*(uint32_t *)gpu_flag_mhandle.host_ptr, val);
                    SB();

                    cpu_flag_idx = 0;
                    do {
                        if (cpu_flag_mhandle.mem_loc == MEM_LOC_GPU)
                            gdr_copy_from_mapping(cpu_flag_mhandle.mh, &val, &((uint32_t *)cpu_flag_mhandle.host_ptr)[cpu_flag_idx], sizeof(uint32_t));
                        else
                            val = READ_ONCE(((uint32_t *)cpu_flag_mhandle.host_ptr)[cpu_flag_idx]);
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

                lat_us = time_diff(beg, end) / (double)num_iters;
                break;
            }
            case BENCHMARK_MODE_GPU_PRODUCE_CPU_CONSUME: {
                uint64_t gpu_time_beg = 0, gpu_time_end = 0;
                // Notify GPU that CPU is ready so that GPU can start the timer.
                if (gpu_flag_loc == MEM_LOC_GPU)
                    gdr_copy_to_mapping(gpu_flag_mhandle.mh, gpu_flag_mhandle.host_ptr, &val, sizeof(gpu_flag_mhandle.size));
                else
                    WRITE_ONCE(*(uint32_t *)gpu_flag_mhandle.host_ptr, val);
                SB();

                clock_gettime(MYCLOCK, &beg);
                while (i <= num_iters) {
                    cpu_flag_idx = 0;
                    do {
                        if (cpu_flag_mhandle.mem_loc == MEM_LOC_GPU)
                            gdr_copy_from_mapping(cpu_flag_mhandle.mh, &val, &((uint32_t *)cpu_flag_mhandle.host_ptr)[cpu_flag_idx], sizeof(uint32_t));
                        else
                            val = READ_ONCE(((uint32_t *)cpu_flag_mhandle.host_ptr)[cpu_flag_idx]);
                        if (val == i + 1)
                            ++cpu_flag_idx;
                        else
                            check_timeout(beg, timeout_us);
                    }
                    while (cpu_flag_idx < num_blocks);
                    LB();

                    if (gpu_flag_loc == MEM_LOC_GPU)
                        gdr_copy_to_mapping(gpu_flag_mhandle.mh, gpu_flag_mhandle.host_ptr, &val, sizeof(gpu_flag_mhandle.size));
                    else
                        WRITE_ONCE(*(uint32_t *)gpu_flag_mhandle.host_ptr, val);
                    SB();

                    i = val;
                }

                ASSERTDRV(cuStreamSynchronize(0));

                ASSERTDRV(cuMemcpyDtoH(&gpu_time_beg, gpu_beg_mhandle.gpu_ptr, sizeof(uint64_t)));
                ASSERTDRV(cuMemcpyDtoH(&gpu_time_end, gpu_end_mhandle.gpu_ptr, sizeof(uint64_t)));

                lat_us = ((gpu_time_end - gpu_time_beg) / 1000.0) / (double)num_iters;
                break;
            }
            default:
                cout << "ERROR: Unrecognized mode." << endl;
                exit(EXIT_FAILURE);
        }

        ASSERTDRV(cuStreamSynchronize(0));

        cout << "Round-trip latency per iteration is " << lat_us << " us" << endl;

        gh_mem_free(&gpu_flag_mhandle);
        gh_mem_free(&cpu_flag_mhandle);
        gh_mem_free(&gpu_beg_mhandle);
        gh_mem_free(&gpu_end_mhandle);
        if (process_data) {
            gh_mem_free(&data_buf_mhandle);
            gh_mem_free(&init_buf_mhandle);
        }
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);

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
