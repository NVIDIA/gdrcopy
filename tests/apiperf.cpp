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
int num_runs         = 1;
const int max_num_buckets = 100;
int num_buckets = max_num_buckets;
size_t _size = (size_t)1 << 24;
int dev_id = 0;
gdr_map_flags_t map_type_flag = GDR_MAP_FLAG_DEFAULT;

// Per-size median latency of each measured API for a single run.
struct apiperf_result {
    size_t size;
    double pin;
    double map;
    double get_info;
    double unmap;
    double unpin;
};

static void print_run_aggregate_results(const std::vector<apiperf_result> &results, int num_runs)
{
    std::map<size_t, std::vector<apiperf_result> > by_size;
    for (size_t i = 0; i < results.size(); i++)
        by_size[results[i].size].push_back(results[i]);

    for (std::map<size_t, std::vector<apiperf_result> >::iterator it = by_size.begin();
         it != by_size.end(); ++it) {
        std::vector<double> pin_v, map_v, info_v, unmap_v, unpin_v;
        for (size_t i = 0; i < it->second.size(); i++) {
            pin_v.push_back(it->second[i].pin);
            map_v.push_back(it->second[i].map);
            info_v.push_back(it->second[i].get_info);
            unmap_v.push_back(it->second[i].unmap);
            unpin_v.push_back(it->second[i].unpin);
        }
        cout << endl << "Aggregate latency across " << num_runs << " runs, size=" << it->first << endl;
        print_aggregate_stats("pin latency", calc_aggregate_stats(pin_v), " us");
        print_aggregate_stats("map latency", calc_aggregate_stats(map_v), " us");
        print_aggregate_stats("get_info latency", calc_aggregate_stats(info_v), " us");
        print_aggregate_stats("unmap latency", calc_aggregate_stats(unmap_v), " us");
        print_aggregate_stats("unpin latency", calc_aggregate_stats(unpin_v), " us");
    }
}

void print_usage(const char *path)
{
    cout << "Usage: " << path << " [-h][-s <max-size>][-d <gpu>][-n <iters>][-w <iters>][-R <runs>][-a <fn>]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "   -h              Print this help text" << endl;
    cout << "   -s <max-size>   Max buffer size to benchmark (default: " << _size << ")" << endl;
    cout << "   -d <gpu>        GPU ID (default: " << dev_id << ")" << endl;
    cout << "   -n <iters>      Number of benchmark iterations (default: " << num_iters << ")" << endl;
    cout << "   -w <iters>      Number of warm-up iterations (default: " << num_warmup_iters << ")" << endl;
    cout << "   -R <runs>       Number of independent repeat runs (default: " << num_runs << ")" << endl;
    cout << "   -a <fn>         GPU buffer allocation function (default: cuMemAlloc)" << endl;
    cout << "                       Choices: cuMemAlloc, cuMemCreate" << endl;
    cout << "   -M <mapping_type>   Request mapping type (choices: default, wc, cache, device)" << endl;
}

void run_test(CUdeviceptr d_A, size_t size, std::vector<apiperf_result> *results = NULL)
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
    // per-bucket average latency of each measured API
    double pin_bkt[max_num_buckets];
    double map_bkt[max_num_buckets];
    double inf_bkt[max_num_buckets];
    double unmap_bkt[max_num_buckets];
    double unpin_bkt[max_num_buckets];

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
            int lat_count = 0;
            int bucket_iters = num_iters / num_buckets;
            size_t actual_pin_size;
            double min_lat, max_lat;
            min_lat = -1;
            max_lat = -1;
            actual_pin_size = PAGE_ROUND_UP(pin_request_size, GPU_PAGE_SIZE);

            for (iter = 0; iter < num_warmup_iters; ++iter) {

                ASSERT_EQ(gdr_pin_buffer(g, d_A, actual_pin_size, 0, 0, &mh), 0);
                ASSERT_NEQ(mh, null_mh);

                void *map_d_ptr  = NULL;
                ASSERT_EQ(gdr_map_v2(g, mh, &map_d_ptr, actual_pin_size, map_type_flag), 0);

                gdr_info_t info;
                ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
                ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, actual_pin_size), 0);
                ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
            }

            for (int bucket = 0; bucket < num_buckets; bucket++) {
                pin_lat_us = 0;
                map_lat_us = 0;
                unpin_lat_us = 0;
                unmap_lat_us = 0;
                inf_lat_us = 0;
                for (iter = 0; iter < bucket_iters; ++iter) {

                    clock_gettime(MYCLOCK, &beg);
                    ASSERT_EQ(gdr_pin_buffer(g, d_A, actual_pin_size, 0, 0, &mh), 0);
                    clock_gettime(MYCLOCK, &end);
                    delta_lat_us = time_diff(beg, end);
                    pin_lat_us += delta_lat_us;
                    ASSERT_NEQ(mh, null_mh);
                    lat_arr[lat_count++] = delta_lat_us;
                    min_lat = (min_lat == -1) ? delta_lat_us : ((delta_lat_us < min_lat) ? delta_lat_us : min_lat);
                    max_lat = delta_lat_us > max_lat ? delta_lat_us : max_lat;

                    void *map_d_ptr  = NULL;
                    clock_gettime(MYCLOCK, &beg);
                    ASSERT_EQ(gdr_map_v2(g, mh, &map_d_ptr, actual_pin_size, map_type_flag), 0);
                    clock_gettime(MYCLOCK, &end);
                    delta_lat_us = time_diff(beg, end);
                    map_lat_us += delta_lat_us;

                    gdr_info_t info;
                    clock_gettime(MYCLOCK, &beg);
                    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
                    clock_gettime(MYCLOCK, &end);
                    delta_lat_us = time_diff(beg, end);
                    inf_lat_us += delta_lat_us;

                    clock_gettime(MYCLOCK, &beg);
                    ASSERT_EQ(gdr_unmap(g, mh, map_d_ptr, actual_pin_size), 0);
                    clock_gettime(MYCLOCK, &end);
                    delta_lat_us = time_diff(beg, end);
                    unmap_lat_us += delta_lat_us;

                    clock_gettime(MYCLOCK, &beg);
                    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
                    clock_gettime(MYCLOCK, &end);
                    delta_lat_us = time_diff(beg, end);
                    unpin_lat_us += delta_lat_us;
                }
                pin_bkt[bucket]   = pin_lat_us / bucket_iters;
                map_bkt[bucket]   = map_lat_us / bucket_iters;
                inf_bkt[bucket]   = inf_lat_us / bucket_iters;
                unmap_bkt[bucket] = unmap_lat_us / bucket_iters;
                unpin_bkt[bucket] = unpin_lat_us / bucket_iters;
            }

            sort(pin_bkt, pin_bkt + num_buckets);
            sort(map_bkt, map_bkt + num_buckets);
            sort(inf_bkt, inf_bkt + num_buckets);
            sort(unmap_bkt, unmap_bkt + num_buckets);
            sort(unpin_bkt, unpin_bkt + num_buckets);

            double pin_med = median_sorted(pin_bkt, num_buckets),     pin_min = pin_bkt[0];
            double map_med = median_sorted(map_bkt, num_buckets),     map_min = map_bkt[0];
            double inf_med = median_sorted(inf_bkt, num_buckets),     inf_min = inf_bkt[0];
            double unmap_med = median_sorted(unmap_bkt, num_buckets), unmap_min = unmap_bkt[0];
            double unpin_med = median_sorted(unpin_bkt, num_buckets), unpin_min = unpin_bkt[0];

            printf("Stat\tSize(B)\tpin.Time(us)\tmap.Time(us)\tget_info.Time(us)\tunmap.Time(us)\tunpin.Time(us)\n");
            printf("median\t%zu\t%f\t%f\t%f\t%f\t%f\n",
                    actual_pin_size, pin_med, map_med, inf_med, unmap_med, unpin_med);
            printf("min\t%zu\t%f\t%f\t%f\t%f\t%f\n",
                    actual_pin_size, pin_min, map_min, inf_min, unmap_min, unpin_min);

            if (results != NULL) {
                apiperf_result r;
                r.size = actual_pin_size;
                r.pin = pin_med;
                r.map = map_med;
                r.get_info = inf_med;
                r.unmap = unmap_med;
                r.unpin = unpin_med;
                results->push_back(r);
            }
            pin_request_size <<= 1;

            printf("Histogram of gdr_pin_buffer latency for %ld bytes\n", actual_pin_size);
            print_histogram(lat_arr, lat_count, bin_arr, num_bins, min_lat, max_lat);
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
        c = getopt(argc, argv, "s:d:n:w:R:a:M:h");
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
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
                break;
            default:
                printf("ERROR: invalid option\n");
                exit(EXIT_FAILURE);
        }
    }

    if (num_runs <= 0) {
        fprintf(stderr, "ERROR: num_runs must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (num_iters <= 0) {
        fprintf(stderr, "ERROR: num_iters must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (num_iters > max_num_buckets && num_iters % max_num_buckets != 0) {
        int old_num_iters = num_iters;
        num_iters += max_num_buckets - (num_iters % max_num_buckets);
        cerr << "WARNING: num_iters is not a multiple of " << max_num_buckets
             << ". Increasing num_iters from " << old_num_iters
             << " to " << num_iters << "." << endl;
    }
    num_buckets = (num_iters > max_num_buckets) ? max_num_buckets : num_iters;

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
    ASSERTDRV(galloc_fn(&mhandle, size, true, true, false, 0));
    d_A = mhandle.ptr;
    cout << "device ptr: 0x" << hex << d_A << dec << endl;
    cout << "allocated size: " << size << endl;

    if (num_runs == 1) {
        run_test(d_A, size);
    } else {
        std::vector<apiperf_result> results;
        for (int run = 0; run < num_runs; run++) {
            cout << endl << "starting run " << (run + 1) << " of " << num_runs << endl;
            run_test(d_A, size, &results);
        }
        print_run_aggregate_results(results, num_runs);
    }

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
