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

#ifndef __GDRAPI_H__
#define __GDRAPI_H__

#include <stdint.h> // for standard [u]intX_t types
#include <stddef.h>

#define GDR_API_MAJOR_VERSION    1
#define GDR_API_MINOR_VERSION    4
#define GDR_API_VERSION          ((GDR_API_MAJOR_VERSION << 16) | GDR_API_MINOR_VERSION)


#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

/*
 * GDRCopy, a low-latency GPU memory copy library (and a kernel-mode
 * driver) based on NVIDIA GPUDirect RDMA technology.
 *
 * supported environment variables:
 *
 * - GDRCOPY_ENABLE_LOGGING, if defined logging is enabled, default is
 *   disabled.
 *
 * - GDRCOPY_LOG_LEVEL, overrides log threshold, default is to print errors
 *   only.
 */

#ifdef __cplusplus
extern "C" {
#endif

struct gdr;
typedef struct gdr *gdr_t;

// Initialize the library, e.g. by opening a connection to the kernel-mode
// driver. Returns an handle to the library state object.
gdr_t gdr_open();

// Destroy library state object, e.g. it closes the connection to kernel-mode
// driver.
int gdr_close(gdr_t g);

typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

// Map device memory buffer on GPU BAR1, returning an handle.
// Memory is still not accessible to user-space.
int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);

// Unmap the handle. 
//
// If there exists a corresponding user-space mapping, gdr_unmap should be
// called before this one.
int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);

// flag is set when the kernel callback (relative to the
// nvidia_p2p_get_pages) gets invoked, e.g. cuMemFree() before
// gdr_unpin_buffer.
int gdr_get_callback_flag(gdr_t g, gdr_mh_t handle, int *flag);

// After pinning, info struct contains details of the mapped area.  
//
// Note that both info->va and info->mapped_size might be different from
// the original address passed to gdr_pin_buffer due to aligning happening
// in the kernel-mode driver
struct gdr_info {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    uint32_t mapped;
    uint32_t wc_mapping;
};
typedef struct gdr_info gdr_info_t;
int gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info);

// create a user-space mapping for the BAR1 info, length is bar1->size
// above.
//
// WARNING: the BAR physical address will be aligned to the page size
// before being mapped in user-space, so the pointer returned might be
// affected by an offset. gdr_get_info can be used to calculate that
// offset.
int gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);

// get rid of a user-space mapping.
// First invoke gdr_unmap() then gdr_unpin_buffer().
int gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);

// gpubar_ptr is a user-space virtual address, i.e. one returned by gdr_map()
int gdr_copy_to_bar(void  *gpubar_ptr, const void *cpumem_ptr, size_t size);
int gdr_copy_from_bar(void *cpumem_ptr, const void *gpubar_ptr, size_t size);


#ifdef __cplusplus
}
#endif

#endif // __GDRAPI_H__
