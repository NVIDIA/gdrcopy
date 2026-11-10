/*
 * Copyright (c) 2014-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdbool>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

using namespace gdrcopy::test;

// manually tuned...
int num_write_iters = 10000;
int num_read_iters  = 100;
int num_runs = 1;
const int max_num_buckets = 100;
int num_write_buckets = max_num_buckets;
int num_read_buckets = max_num_buckets;
size_t _size = 128*1024;
size_t copy_offset = 0;
int dev_id = 0;
bool no_store_fence = false;
gdr_map_flags_t map_type_flag = GDR_MAP_FLAG_DEFAULT;
uint32_t copy_flags = GDR_COPY_FLAG_DEFAULT;
std::vector<size_t> copy_size;
bool use_locality_domain = false;
int locality_domain_id = 0;
bool use_force_pcie = false;

struct bw_result {
    double median;
    double min;
};

struct copybw_result {
    size_t size;
    bw_result write;
    bw_result read;
};

void print_usage(const char *path)
{
    cout << "Usage: " << path << " [-h][-l][-F][-P][-s <size>][-c <size>][-o <offset>][-d <gpu>][-w <iters>][-r <iters>][-R <runs>][-a <fn>][-M <mapping_type>]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h              Print this help text" << endl;
    cout << "   -s <size>       Buffer allocation size (default: " << _size << ")" << endl;
    cout << "   -c <size>       Copy size (default: " << _size << ", if -l is not set. if -l option is set, -c is used to dictate upper bound for run copy size.)" << endl;
    cout << "   -o <offset>     Copy offset (default: " << copy_offset << ")" << endl;
    cout << "   -l              Run copy sizes from 1B to ROUND_DOWN_POW_2(copy_size), where copy_size is dictated by -c option" << endl;
    cout << "   -d <gpu>        GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -w <iters>      Number of write iterations (default: " << num_write_iters << ")" << endl;
    cout << "   -r <iters>      Number of read iterations (default: " << num_read_iters << ")" << endl;
    cout << "   -n <locality-domain-id>    Locality domain ID for GPU memory locality domain" << endl;
    cout << "                           This option is only supported with cuMemCreate" << endl;
    cout << "   -R <runs>       Number of independent repeat runs (default: " << num_runs << ")" << endl;
    cout << "   -a <fn>         GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                       Choices: cuMemAlloc, cuMemCreate" << endl;
    cout << "   -M <mapping_type>   Request mapping type (choices: default, wc, cache, device)" << endl;
    cout << "   -F              Disable store fences via the v2 copy APIs for higher throughput (default: no)" << endl;
    cout << "   -P              Use GDR_PIN_FLAG_FORCE_PCIE flag (forces WC mapping, cannot be combined with -M cache or -M device)" << endl;
}

static void print_run_aggregate_results(const std::vector<std::vector<copybw_result> > &run_results)
{
    if (run_results.empty())
        return;

    for (size_t size_index = 0; size_index < copy_size.size(); size_index++) {
        std::vector<double> write_medians;
        std::vector<double> read_medians;
        size_t size = 0;

        for (size_t run = 0; run < run_results.size(); run++) {
            if (size_index >= run_results[run].size())
                continue;

            size = run_results[run][size_index].size;
            write_medians.push_back(run_results[run][size_index].write.median);
            read_medians.push_back(run_results[run][size_index].read.median);
        }

        if (write_medians.empty() || read_medians.empty())
            continue;

        cout << endl << "Aggregate BW across " << run_results.size()
             << " runs, size=" << size
             << " offset=" << copy_offset << endl;
        print_aggregate_stats("Write BW", calc_aggregate_stats(write_medians), " MB/s");
        print_aggregate_stats("Read BW", calc_aggregate_stats(read_medians), " MB/s");
    }
}

std::vector<copybw_result> run_test(CUdeviceptr d_A, size_t size)
{
    std::vector<copybw_result> results;
    uint32_t *init_buf = NULL;
    ASSERTDRV(cuMemAllocHost((void **)&init_buf, size));
    ASSERT_NEQ(init_buf, (void*)0);
    init_hbuf_walking_bit(init_buf, size);

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    BEGIN_CHECK {
        int pin_flags = GDR_PIN_FLAG_DEFAULT;
        if (use_force_pcie) {
            pin_flags |= GDR_PIN_FLAG_FORCE_PCIE;
        }
        ASSERT_EQ(gdr_pin_buffer_v2(g, d_A, size, pin_flags, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        void *map_d_ptr  = NULL;
        ASSERT_EQ(gdr_map_v2(g, mh, &map_d_ptr, size, map_type_flag), 0);
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
        cout << "store fences: " << (no_store_fence ? "disabled (using v2 API)" : "enabled") << endl;
        double bw_MBps[max_num_buckets];

        for (int i = 0; i < copy_size.size(); i++)
        {
            copybw_result result;
            result.size = copy_size[i];

            // copy to GPU benchmark
            cout << "writing test, size=" << copy_size[i] << " offset=" << copy_offset << " num_iters=" << num_write_iters << endl;
            struct timespec beg, end;
            for (int bucket = 0; bucket < num_write_buckets; bucket++) {
                int bucket_iters = num_write_iters / num_write_buckets;
                clock_gettime(MYCLOCK, &beg);
                for (int iter = 0; iter < bucket_iters; ++iter)
                    gdr_copy_to_mapping_v2(mh, buf_ptr + copy_offset / 4, init_buf, copy_size[i], copy_flags);
                clock_gettime(MYCLOCK, &end);

                double byte_count = (double)copy_size[i] * bucket_iters;
                double dt_us = time_diff(beg, end);
                double Bps = byte_count / dt_us * 1e6;
                bw_MBps[bucket] = Bps / 1024.0 / 1024.0;
            }
            aggregate_stats write_stats = calc_aggregate_stats(bw_MBps, bw_MBps + num_write_buckets);
            result.write.median = write_stats.median;
            result.write.min = write_stats.min;
            cout << "write BW: median " << result.write.median << "MB/s, min " << result.write.min << "MB/s" << endl;

            if(no_store_fence)
                gdr_copy_fence(mh, GDR_COPY_FLAG_WRITE_FENCE);
            compare_buf(init_buf, buf_ptr + copy_offset / 4, copy_size[i]);

            // copy from GPU benchmark
            cout << "reading test, size=" << copy_size[i] << " offset=" << copy_offset << " num_iters=" << num_read_iters << endl;
            for (int bucket = 0; bucket < num_read_buckets; bucket++) {
                int bucket_iters = num_read_iters / num_read_buckets;
                clock_gettime(MYCLOCK, &beg);
                for (int iter = 0; iter < bucket_iters; ++iter)
                    gdr_copy_from_mapping_v2(mh, init_buf, buf_ptr + copy_offset / 4, copy_size[i], copy_flags);
                clock_gettime(MYCLOCK, &end);

                double byte_count = (double)copy_size[i] * bucket_iters;
                double dt_us = time_diff(beg, end);
                double Bps = byte_count / dt_us * 1e6;
                bw_MBps[bucket] = Bps / 1024.0 / 1024.0;
            }
            aggregate_stats read_stats = calc_aggregate_stats(bw_MBps, bw_MBps + num_read_buckets);
            result.read.median = read_stats.median;
            result.read.min = read_stats.min;
            cout << "read BW: median " << result.read.median << "MB/s, min " << result.read.min << "MB/s" << endl;
            results.push_back(result);
        }

        cout << "unmapping buffer" << endl;
        ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, size), 0);

        cout << "unpinning buffer" << endl;
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;

    cout << "closing gdrdrv" << endl;
    ASSERT_EQ(gdr_close(g), 0);
    ASSERTDRV(cuMemFreeHost(init_buf));

    return results;
}

int main(int argc, char *argv[])
{
    gpu_memalloc_fn_t galloc_fn = gpu_mem_alloc;
    gpu_memfree_fn_t gfree_fn = gpu_mem_free;
    size_t copy_size_upper_bound = 0;
    bool copy_size_range = false;

    while (1) {
        int c;
        c = getopt(argc, argv, "s:d:o:c:w:r:R:a:n:M:hlFP");
        if (c == -1)
            break;

        switch (c) {
        case 's':
            _size = strtol(optarg, NULL, 0);
            break;
        case 'c':
	    copy_size_upper_bound = strtol(optarg, NULL, 0);
            break;
        case 'l':
          copy_size_range = true;
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
        case 'R':
            num_runs = strtol(optarg, NULL, 0);
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
        case 'n':
#ifndef HAVE_DEVICE_LOCALITY_DOMAIN
            cerr << "Locality domain allocation requires CUDA toolkit 13.4 or later. Current version: " << CUDA_VERSION << endl;
            exit(EXIT_FAILURE);
#else
            use_locality_domain = true;
            locality_domain_id = strtol(optarg, NULL, 0);
#endif
            break;
        case 'M':
            if (strcmp(optarg, "default") == 0) {
                map_type_flag = GDR_MAP_FLAG_DEFAULT;
            } else if (strcmp(optarg, "wc") == 0) {
                map_type_flag = GDR_MAP_FLAG_REQ_WC_MAPPING;
            } else if (strcmp(optarg, "cache") == 0) {
                map_type_flag = GDR_MAP_FLAG_REQ_CACHE_MAPPING;
            } else if (strcmp(optarg, "device") == 0) {
                map_type_flag = GDR_MAP_FLAG_REQ_DEVICE_MAPPING;
            } else {
                cerr << "ERROR: invalid mapping_type '" << optarg
                     << "'. Valid options: default, wc, cache, device." << endl;
                exit(EXIT_FAILURE);
            }
            break;
        case 'F':
            no_store_fence = true;
            copy_flags = GDR_FLAG_UNSET(GDR_COPY_FLAG_DEFAULT, GDR_COPY_FLAG_WRITE_FENCE);
            break;
        case 'P':
            use_force_pcie = true;
            if (map_type_flag == GDR_MAP_FLAG_REQ_CACHE_MAPPING || map_type_flag == GDR_MAP_FLAG_REQ_DEVICE_MAPPING) {
                cerr << "ERROR: -P forces a WC mapping, which cannot satisfy -M cache or -M device." << endl;
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

    copy_size.clear();
    if (!copy_size_range) {
        /* if -l option is not passed */
	if (copy_size_upper_bound == 0) {
		copy_size.push_back(_size);
	} else {
		copy_size.push_back(copy_size_upper_bound);
	}
    } else {
        /* if -l option is passed */
	size_t upper = (copy_size_upper_bound == 0) ? _size : std::min(copy_size_upper_bound, static_cast<size_t>(_size));
	for (size_t i = 1; i < (ROUND_DOWN_POW_2(upper)); i <<= 1) {
	    copy_size.push_back(i);
        }
    }

    if (use_locality_domain && galloc_fn != gpu_vmm_alloc) {
        cerr << "Locality domain allocation is only supported with cuMemCreate" << endl;
        exit(EXIT_FAILURE);
    }

    if (copy_offset % sizeof(uint32_t) != 0) {
        fprintf(stderr, "ERROR: offset must be multiple of 4 bytes\n");
        exit(EXIT_FAILURE);
    }

    if (num_write_iters <= 0) {
        fprintf(stderr, "ERROR: num_write_iters must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (num_read_iters <= 0) {
        fprintf(stderr, "ERROR: num_read_iters must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (num_runs <= 0) {
        fprintf(stderr, "ERROR: num_runs must be positive\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < copy_size.size(); i++) {
        if (copy_offset + copy_size[i] > _size) {
            fprintf(stderr, "ERROR: offset + copy size run past the end of the buffer\n");
            exit(EXIT_FAILURE);
        }
    }

    if(num_write_iters > max_num_buckets && num_write_iters % max_num_buckets != 0){
        int old_num_write_iters = num_write_iters;
        num_write_iters += max_num_buckets - (num_write_iters % max_num_buckets);
        cerr << "WARNING: num_write_iters is not a multiple of " << max_num_buckets
             << ". Increasing num_write_iters from " << old_num_write_iters
             << " to " << num_write_iters << "." << endl;
    }
    if(num_read_iters > max_num_buckets && num_read_iters % max_num_buckets != 0){
        int old_num_read_iters = num_read_iters;
        num_read_iters += max_num_buckets - (num_read_iters % max_num_buckets);
        cerr << "WARNING: num_read_iters is not a multiple of " << max_num_buckets
             << ". Increasing num_read_iters from " << old_num_read_iters
             << " to " << num_read_iters << "." << endl;
    }
    num_write_buckets = (num_write_iters > max_num_buckets) ? max_num_buckets : num_write_iters;
    num_read_buckets = (num_read_iters > max_num_buckets) ? max_num_buckets : num_read_iters;

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
    ASSERTDRV(galloc_fn(&mhandle, size, true, true, use_locality_domain, locality_domain_id));
    d_A = mhandle.ptr;
    cout << "device ptr: " << hex << d_A << dec << endl;
    cout << "use force pcie: " << (use_force_pcie ? "yes" : "no") << endl;

    if (num_runs == 1) {
        run_test(d_A, size);
    } else {
        std::vector<std::vector<copybw_result> > run_results;
        for (int run = 0; run < num_runs; run++) {
            cout << endl << "starting run " << (run + 1) << " of " << num_runs << endl;
            run_results.push_back(run_test(d_A, size));
        }
        print_run_aggregate_results(run_results);
    }

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
