/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <ctype.h>
#include <signal.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <check.h>
#include <errno.h>
#include <sys/queue.h>

using namespace std;

#include "gdrapi.h"
#include "gdrapi_internal.h"
#include "gdrconfig.h"
#include "common.hpp"

using namespace gdrcopy::test;

volatile bool expecting_exception_signal = false;

void exception_signal_handle(int sig)
{
    if (expecting_exception_signal) {
        print_dbg("Get signal %d as expected\n", sig);
        exit(EXIT_SUCCESS);
    }
    print_dbg("Unexpectedly get exception signal");
}

void init_cuda(int dev_id)
{
    CUdevice dev;
    CUcontext dev_ctx;
    ASSERTDRV(cuInit(0));
    ASSERTDRV(cuDeviceGet(&dev, dev_id));

    ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
    ASSERTDRV(cuCtxSetCurrent(dev_ctx));

    ASSERT_EQ(check_gdr_support(dev), true);
}

void finalize_cuda(int dev_id)
{
    CUdevice dev;
    ASSERTDRV(cuDeviceGet(&dev, dev_id));
    ASSERTDRV(cuDevicePrimaryCtxRelease(dev));
}

typedef void (*filter_fn_t)();

void null_filter()
{
    // NO-OP.
}

#if CUDA_VERSION >= 11000
/**
 * Waive the test if VMM is not supported.
 * Must be called after init_cuda.
 */
void vmm_filter()
{
    int version;
    ASSERTDRV(cuDriverGetVersion(&version));
    if (version < 11000)
        exit(EXIT_WAIVED);
}
#else
void vmm_filter()
{
    exit(EXIT_WAIVED);
}
#endif

/**
 * Sends given file descriptior via given socket
 *
 * @param socket to be used for fd sending
 * @param fd to be sent
 * @return sendmsg result
 *
 * @note socket should be (PF_UNIX, SOCK_DGRAM)
 */
int sendfd(int socket, int fd)
{
    char dummy = '$';
    struct msghdr msg;
    struct iovec iov;

    char cmsgbuf[CMSG_SPACE(sizeof(int))];

    iov.iov_base = &dummy;
    iov.iov_len = sizeof(dummy);

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_flags = 0;
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = CMSG_LEN(sizeof(int));

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));

    *(int*) CMSG_DATA(cmsg) = fd;

    int ret = sendmsg(socket, &msg, 0);

    if (ret == -1) {
        print_dbg("sendmsg failed with %s", strerror(errno));
    }

    return ret;
}

/**
 * Receives file descriptor using given socket
 *
 * @param socket to be used for fd recepion
 * @return received file descriptor; -1 if failed
 *
 * @note socket should be (PF_UNIX, SOCK_DGRAM)
 */
int recvfd(int socket) 
{
    int len;
    int fd;
    char buf[1];
    struct iovec iov;
    struct msghdr msg;
    struct cmsghdr *cmsg;
    char cms[CMSG_SPACE(sizeof(int))];

    iov.iov_base = buf;
    iov.iov_len = sizeof(buf);

    msg.msg_name = 0;
    msg.msg_namelen = 0;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_flags = 0;
    msg.msg_control = (caddr_t) cms;
    msg.msg_controllen = sizeof cms;

    len = recvmsg(socket, &msg, 0);

    if (len < 0) {
        print_dbg("recvmsg failed with %s", strerror(errno));
        return -1;
    }

    if (len == 0) {
        print_dbg("recvmsg failed no data");
        return -1;
    }

    cmsg = CMSG_FIRSTHDR(&msg);
    memmove(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void basic()
{
    expecting_exception_signal = false;
    MB();

    init_cuda(0);
    filter_fn();

    const size_t _size = 256*1024+16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    print_dbg("buffer size: %zu\n", size);
    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh = null_mh;
    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(gfree_fn(&mhandle));

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(basic_cumemalloc)
{
    basic<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(basic_vmmalloc)
{
    basic<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(basic_with_tokens)
{
    expecting_exception_signal = false;
    MB();

    init_cuda(0);

    const size_t _size = 256*1024+16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    print_dbg("buffer size: %zu\n", size);

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens = {0,0};

    // Token does not work with cuMemCreate
    ASSERTDRV(gpu_mem_alloc(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuPointerGetAttribute(&tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, d_A));

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh = null_mh;
    CUdeviceptr d_ptr = d_A;

    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, tokens.p2pToken, tokens.vaSpaceToken, &mh), 0);
    ASSERT_NEQ(mh, null_mh);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(gpu_mem_free(&mhandle));

    finalize_cuda(0);
}
END_GDRCOPY_TEST

/**
 * This unit test ensures that gdrcopy returns error when trying to map
 * unaligned addresses. In addition, it tests that mapping hand-aligned
 * addresses by users are successful.
 *
 * cuMemCreate + cuMemMap always return an aligned address. So, this test is
 * for cuMemAlloc only.
 *
 */
BEGIN_GDRCOPY_TEST(basic_unaligned_mapping)
{
    expecting_exception_signal = false;
    MB();

    init_cuda(0);

    // Allocate for a few bytes so that cuMemAlloc returns an unaligned address
    // in the next allocation. This behavior is observed in GPU Driver 410 and
    // above.
    const size_t fa_size = 4;
    CUdeviceptr d_fa;
    gpu_mem_handle_t fa_mhandle;
    ASSERTDRV(gpu_mem_alloc(&fa_mhandle, fa_size, true, true));
    d_fa = fa_mhandle.ptr;
    print_dbg("First allocation: d_fa=0x%llx, size=%zu\n", d_fa, fa_size);

    const size_t A_size = GPU_PAGE_SIZE + sizeof(int);

    const int retry = 10;
    int cnt = 0;

    CUdeviceptr d_A, d_A_boundary;
    gpu_mem_handle_t A_mhandle[retry];

    // Try until we get an unaligned address. Give up after cnt times.
    for (cnt = 0; cnt < retry; ++cnt) {
        ASSERTDRV(gpu_mem_alloc(&A_mhandle[cnt], A_size, false, true));
        d_A = A_mhandle[cnt].ptr;
        d_A_boundary = d_A & GPU_PAGE_MASK;
        if (d_A != d_A_boundary) {
            ++cnt;
            break;
        }
    }
    print_dbg("Second allocation: d_A=0x%llx, size=%zu, GPU-page-boundary 0x%llx\n", d_A, A_size, d_A_boundary);
    if (d_A == d_A_boundary) {
        print_dbg("d_A is aligned. Waiving this test.\n");
        for (int i = 0; i < cnt; ++i)
            ASSERTDRV(gpu_mem_free(&A_mhandle[i]));

        exit(EXIT_WAIVED);
    }
    print_dbg("d_A is unaligned\n");

    gdr_t g = gdr_open_safe();

    // Try mapping with unaligned address. This should fail.
    print_dbg("Try mapping d_A as is.\n");
    gdr_mh_t A_mh = null_mh;

    ASSERT_EQ(gdr_pin_buffer(g, d_A, A_size, 0, 0, &A_mh), 0);
    ASSERT_NEQ(A_mh, null_mh);

    void *A_bar_ptr  = NULL;
    // Expect gdr_map to fail with unaligned address
    ASSERT_NEQ(gdr_map(g, A_mh, &A_bar_ptr, A_size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, A_mh), 0);
    print_dbg("Mapping d_A failed as expected.\n");

    print_dbg("Align d_A and try mapping it again.\n");
    // In order to align d_A, we move to the next GPU page. The reason is that
    // the first GPU page may belong to another allocation.
    CUdeviceptr d_aligned_A = (d_A + GPU_PAGE_SIZE) & GPU_PAGE_MASK;
    off_t aligned_A_offset = d_aligned_A - d_A;
    size_t aligned_A_size = A_size - aligned_A_offset;

    print_dbg("Pin and map aligned address: d_aligned_A=0x%llx, offset=%lld, size=%zu\n", d_aligned_A, aligned_A_offset, aligned_A_size);

    gdr_mh_t aligned_A_mh = null_mh;
    void *aligned_A_bar_ptr = NULL;
    ASSERT_EQ(gdr_pin_buffer(g, d_aligned_A, aligned_A_size, 0, 0, &aligned_A_mh), 0);
    ASSERT_NEQ(aligned_A_mh, null_mh);
    // expect gdr_map to success
    ASSERT_EQ(gdr_map(g, aligned_A_mh, &aligned_A_bar_ptr, aligned_A_size), 0);

    // Test accessing the mapping
    int *aligned_A_map_ptr = (int *)aligned_A_bar_ptr;
    aligned_A_map_ptr[0] = 7;

    // The first allocation and d_A should share a GPU page. We should make
    // sure that freeing the first allocation would not accidentally unmap
    // d_aligned_A as the d_aligned_A mapping starts from the next GPU page.
    gdr_mh_t fa_mh = null_mh;
    ASSERT_EQ(gdr_pin_buffer(g, d_fa, fa_size, 0, 0, &fa_mh), 0);
    ASSERT_NEQ(fa_mh, null_mh);

    void *fa_bar_ptr = NULL;
    ASSERT_EQ(gdr_map(g, fa_mh, &fa_bar_ptr, fa_size), 0);

    ASSERTDRV(gpu_mem_free(&fa_mhandle));

    // Test accessing aligned_A_map_ptr again. This should not cause segmentation fault.
    aligned_A_map_ptr[0] = 9;

    ASSERT_EQ(gdr_unpin_buffer(g, aligned_A_mh), 0);
    ASSERT_EQ(gdr_close(g), 0);

    for (int i = 0; i < cnt; ++i)
        ASSERTDRV(gpu_mem_free(&A_mhandle[i]));

    finalize_cuda(0);
}
END_GDRCOPY_TEST

template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void data_validation()
{
    expecting_exception_signal = false;
    MB();

    init_cuda(0);
    filter_fn();

    const size_t _size = 256*1024+16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    print_dbg("buffer size: %zu\n", size);
    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0xA5, size));
    ASSERTDRV(cuCtxSynchronize());

    uint32_t *init_buf = new uint32_t[size];
    uint32_t *copy_buf = new uint32_t[size];

    init_hbuf_walking_bit(init_buf, size);
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    ASSERT(!info.mapped);

    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    ASSERT(info.mapped);
    int off = d_ptr - info.va;
    print_dbg("off: %d\n", off);

    uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);

    print_dbg("check 1: MMIO CPU initialization + read back via cuMemcpy D->H\n");
    init_hbuf_walking_bit(buf_ptr, size);
    ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
    ASSERT_EQ(compare_buf(init_buf, copy_buf, size), 0);
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf));
    ASSERTDRV(cuMemsetD8(d_A, 0xA5, size));
    ASSERTDRV(cuCtxSynchronize());

    print_dbg("check 2: gdr_copy_to_bar() + read back via cuMemcpy D->H\n");
    gdr_copy_to_mapping(mh, buf_ptr, init_buf, size);
    ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
    ASSERT_EQ(compare_buf(init_buf, copy_buf, size), 0);
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf));
    ASSERTDRV(cuMemsetD8(d_A, 0xA5, size));
    ASSERTDRV(cuCtxSynchronize());

    print_dbg("check 3: gdr_copy_to_bar() + read back via gdr_copy_from_bar()\n");
    gdr_copy_to_mapping(mh, buf_ptr, init_buf, size);
    gdr_copy_from_mapping(mh, copy_buf, buf_ptr, size);
    ASSERT_EQ(compare_buf(init_buf, copy_buf, size), 0);
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf));
    ASSERTDRV(cuMemsetD8(d_A, 0xA5, size));
    ASSERTDRV(cuCtxSynchronize());

    int extra_dwords = 5;
    int extra_off = extra_dwords * sizeof(uint32_t);
    print_dbg("check 4: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d dwords offset\n", extra_dwords);
    gdr_copy_to_mapping(mh, buf_ptr + extra_dwords, init_buf, size - extra_off);
    gdr_copy_from_mapping(mh, copy_buf, buf_ptr + extra_dwords, size - extra_off);
    ASSERT_EQ(compare_buf(init_buf, copy_buf, size - extra_off), 0);
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf));
    ASSERTDRV(cuMemsetD8(d_A, 0xA5, size));
    ASSERTDRV(cuCtxSynchronize());

    extra_off = 11;
    print_dbg("check 5: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d bytes offset\n", extra_off);
    gdr_copy_to_mapping(mh, (char*)buf_ptr + extra_off, init_buf, size - extra_off);
    gdr_copy_from_mapping(mh, copy_buf, (char*)buf_ptr + extra_off, size - extra_off);
    ASSERT_EQ(compare_buf(init_buf, copy_buf, size - extra_off), 0);

    print_dbg("unmapping\n");
    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    print_dbg("unpinning\n");
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(gfree_fn(&mhandle));

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(data_validation_cumemalloc)
{
    data_validation<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(data_validation_vmmalloc)
{
    data_validation<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * This unit test ensures that accessing to gdr_map'ed region is not possible
 * after gdr_close.
 *
 * Step:
 * 1. Initialize CUDA and gdrcopy
 * 2. Do gdr_map(..., &bar_ptr, ...)
 * 3. Do gdr_close
 * 4. Attempt to access to bar_ptr after 3. should fail
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_access_after_gdr_close()
{
    expecting_exception_signal = false;
    MB();

    struct sigaction act;
    act.sa_handler = exception_signal_handle;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGBUS, &act, 0);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    int mydata = (rand() % 1000) + 1;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    print_dbg("Mapping bar1\n");
    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = d_ptr - info.va;

    volatile int *buf_ptr = (volatile int *)((char *)bar_ptr + off);

    // Write data
    print_dbg("Writing %d into buf_ptr[0]\n", mydata);
    buf_ptr[0] = mydata;

    print_dbg("Calling gdr_close\n");
    ASSERT_EQ(gdr_close(g), 0);

    print_dbg("Trying to read buf_ptr[0] after gdr_close\n");
    expecting_exception_signal = true;
    MB();
    int data_from_buf_ptr = buf_ptr[0];
    MB();
    expecting_exception_signal = false;
    MB();

    ASSERT_NEQ(data_from_buf_ptr, mydata);

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_access_after_gdr_close_cumemalloc)
{
    invalidation_access_after_gdr_close<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_access_after_gdr_close_vmmalloc)
{
    invalidation_access_after_gdr_close<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * This unit test ensures that accessing to gdr_map'ed region is not possible
 * after gpuMemFree.
 *
 * Step:
 * 1. Initialize CUDA and gdrcopy
 * 2. Do gdr_map(..., &bar_ptr, ...)
 * 3. Do gpuMemFree
 * 4. Attempt to access to bar_ptr after 3. should fail
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_access_after_free()
{
    expecting_exception_signal = false;
    MB();

    struct sigaction act;
    act.sa_handler = exception_signal_handle;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGBUS, &act, 0);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    int mydata = (rand() % 1000) + 1;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;
    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    print_dbg("Mapping bar1\n");
    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = d_ptr - info.va;

    volatile int *buf_ptr = (volatile int *)((char *)bar_ptr + off);

    // Write data
    print_dbg("Writing %d into buf_ptr[0]\n", mydata);
    buf_ptr[0] = mydata;

    print_dbg("Calling gpuMemFree\n");
    ASSERTDRV(gfree_fn(&mhandle));

    print_dbg("Trying to read buf_ptr[0] after gpuMemFree\n");
    expecting_exception_signal = true;
    MB();
    int data_from_buf_ptr = buf_ptr[0];
    MB();
    expecting_exception_signal = false;
    MB();

    ASSERT_NEQ(data_from_buf_ptr, mydata);

    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    ASSERT_EQ(gdr_close(g), 0);

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_access_after_free_cumemalloc)
{
    invalidation_access_after_free<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_access_after_free_vmmalloc)
{
    invalidation_access_after_free<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST


/**
 * This unit test ensures that gpuMemFree destroys only the mapping it is
 * corresponding to.
 *
 * Step:
 * 1. Initialize CUDA and gdrcopy
 * 2. cuMemAlloc(&d_A, ...); cuMemAlloc(&d_B, ...)
 * 3. Do gdr_map(..., &bar_ptr_A, ...) of d_A
 * 4. Do gdr_map(..., &bar_ptr_B, ...) of d_B
 * 5. Do gpuMemFree(d_A)
 * 6. Verify that bar_ptr_B is still accessible 
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_two_mappings()
{
    expecting_exception_signal = false;
    MB();

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    int mydata = (rand() % 1000) + 1;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A[2];
    gpu_mem_handle_t mhandle[2];

    for (int i = 0; i < 2; ++i) {
        ASSERTDRV(galloc_fn(&mhandle[i], size, true, true));
        d_A[i] = mhandle[i].ptr;
        ASSERTDRV(cuMemsetD8(d_A[i], 0x95, size));
    }
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh[2];

    volatile int *buf_ptr[2];
    void *bar_ptr[2];

    print_dbg("Mapping bar1\n");
    for (int i = 0; i < 2; ++i) {
        CUdeviceptr d_ptr = d_A[i];

        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh[i]), 0);
        ASSERT_NEQ(mh[i], null_mh);

        bar_ptr[i] = NULL;
        ASSERT_EQ(gdr_map(g, mh[i], &bar_ptr[i], size), 0);

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh[i], &info), 0);
        int off = d_ptr - info.va;

        buf_ptr[i] = (volatile int *)((char *)bar_ptr[i] + off);
    }


    // Write data
    print_dbg("Writing data to both mappings %d and %d respectively\n", mydata, mydata + 1);
    buf_ptr[0][0] = mydata;
    buf_ptr[1][0] = mydata + 1;

    print_dbg("Validating that we can read the data back\n");
    ASSERT_EQ(buf_ptr[0][0], mydata);
    ASSERT_EQ(buf_ptr[1][0], mydata + 1);

    print_dbg("gpuMemFree and thus destroying the first mapping\n");
    ASSERTDRV(gfree_fn(&mhandle[0]));

    print_dbg("Trying to read and validate the data from the second mapping after the first mapping has been destroyed\n");
    ASSERT_EQ(buf_ptr[1][0], mydata + 1);

    ASSERTDRV(gfree_fn(&mhandle[1]));

    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(gdr_unmap(g, mh[i], bar_ptr[i], size), 0);
        ASSERT_EQ(gdr_unpin_buffer(g, mh[i]), 0);
    }

    ASSERT_EQ(gdr_close(g), 0);

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_two_mappings_cumemalloc)
{
    invalidation_two_mappings<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_two_mappings_vmmalloc)
{
    invalidation_two_mappings<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * This unit test is intended to check the security hole originated from not
 * doing invalidation correctly. In a nutshell, it ensures that the parent
 * process cannot spy on the child process.
 *
 * Step:
 * 1. Fork the process
 * 2.C Child: Waiting for parent's signal before continue
 *
 * 2.P Parent: Initialize CUDA and gdrcopy
 * 3.P Parent: Do gdr_map then gpuMemFree without gdr_unmap
 * 4.P Parent: Signal child and wait for child's signal
 *
 * 3.C Child: Initialize CUDA and gdrcopy
 * 4.C Child: Do gdr_map, signal parent, and wait for parent's signal
 *
 * 5.P Parent: Check whether it can access to its gdr_map'ed data or not and
 *     compare with the data written by child. If gdrdrv does not handle
 *     invalidation properly, child's data will be leaked to parent.
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_fork_access_after_free()
{
    expecting_exception_signal = false;
    MB();

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ASSERT_NEQ(pipe(filedes_0), -1);
    ASSERT_NEQ(pipe(filedes_1), -1);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    ASSERT(pid >= 0);

    myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    if (pid == 0) {
        close(filedes_0[0]);
        close(filedes_1[1]);

        read_fd = filedes_1[0];
        write_fd = filedes_0[1];

        int cont = 0;

        do {
            print_dbg("%s: waiting for cont signal from parent\n", myname);
            ASSERT_EQ(read(read_fd, &cont, sizeof(int)), sizeof(int));
            print_dbg("%s: receive cont signal %d from parent\n", myname, cont);
        } while (cont != 1);
    }
    else {
        close(filedes_0[1]);
        close(filedes_1[0]);

        read_fd = filedes_0[0];
        write_fd = filedes_1[1];

        struct sigaction act;
        act.sa_handler = exception_signal_handle;
        sigemptyset(&act.sa_mask);
        act.sa_flags = 0;
        sigaction(SIGBUS, &act, 0);
    }

    int mydata = (rand() % 1000) + 1;

    // Make sure that parent's and child's mydata are different.
    // Remember that we do srand before fork.
    if (pid == 0)
        mydata += 10;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = d_ptr - info.va;

    volatile int *buf_ptr = (volatile int *)((char *)bar_ptr + off);

    print_dbg("%s: writing buf_ptr[0] with %d\n", myname, mydata);
    buf_ptr[0] = mydata;

    if (pid == 0) {
        print_dbg("%s: signal parent that I have written\n", myname);
        ASSERT_EQ(write(write_fd, &mydata, sizeof(int)), sizeof(int));

        int cont = 0;
        print_dbg("%s: waiting for signal from parent before calling gpuMemFree\n", myname);
        do {
            ASSERT_NEQ(read(read_fd, &cont, sizeof(int)), -1);
        } while (cont != 1);
    }

    print_dbg("%s: read buf_ptr[0] before gpuMemFree get %d\n", myname, buf_ptr[0]);

    print_dbg("%s: calling gpuMemFree\n", myname);
    ASSERTDRV(gfree_fn(&mhandle));

    if (pid > 0) {
        int msg = 1;
        ASSERT_EQ(write(write_fd, &msg, sizeof(int)), sizeof(int));
        int child_data = 0;
        print_dbg("%s: waiting for child write signal\n", myname);
        do {
            ASSERT_EQ(read(read_fd, &child_data, sizeof(int)), sizeof(int));
        } while (child_data == 0);

        print_dbg("%s: trying to read buf_ptr[0]\n", myname);
        expecting_exception_signal = true;
        MB();
        int data_from_buf_ptr = buf_ptr[0];
        MB();
        expecting_exception_signal = false;
        MB();

        print_dbg("%s: read buf_ptr[0] after child write get %d\n", myname, data_from_buf_ptr);
        print_dbg("%s: child data is %d\n", myname, child_data);
        ASSERT_EQ(write(write_fd, &msg, sizeof(int)), sizeof(int));
        ASSERT_NEQ(child_data, data_from_buf_ptr);
    }

    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

    ASSERT_EQ(gdr_close(g), 0);

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_fork_access_after_free_cumemalloc)
{
    invalidation_fork_access_after_free<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_fork_access_after_free_vmmalloc)
{
    invalidation_fork_access_after_free<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * This unit test makes sure that child processes cannot spy on the parent
 * process if the parent does fork without doing gdr_unmap first.
 *
 * Step:
 * 1. Initilize CUDA and gdrcopy
 * 2. Do gdr_map
 * 3. Fork the process
 *
 * 4.P Parent: Waiting for child to exit
 * 
 * 4.C Child: Attempt to access the gdr_map'ed data and compare with what
 *     parent writes into that region. If gdrdrv does not invalidate the
 *     mapping correctly, child can spy on parent.
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_fork_after_gdr_map()
{
    expecting_exception_signal = false;
    MB();

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ASSERT_NEQ(pipe(filedes_0), -1);
    ASSERT_NEQ(pipe(filedes_1), -1);

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = d_ptr - info.va;

    volatile int *buf_ptr = (volatile int *)((char *)bar_ptr + off);

    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    ASSERT(pid >= 0);

    myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    srand(time(NULL));

    int mynumber = rand() % 1000 + 1;

    if (pid == 0) {
        close(filedes_0[0]);
        close(filedes_1[1]);

        read_fd = filedes_1[0];
        write_fd = filedes_0[1];

        srand(rand());
        int cont = 0;

        do {
            print_dbg("%s: waiting for cont signal from parent\n", myname);
            ASSERT_EQ(read(read_fd, &cont, sizeof(int)), sizeof(int));
            print_dbg("%s: receive cont signal %d from parent\n", myname, cont);
        } while (cont != 1);
    }
    else {
        close(filedes_0[1]);
        close(filedes_1[0]);

        read_fd = filedes_0[0];
        write_fd = filedes_1[1];
    }

    if (pid > 0) {
        print_dbg("%s: writing buf_ptr[0] with %d\n", myname, mynumber);
        buf_ptr[0] = mynumber;
    }

    if (pid == 0) {
        struct sigaction act;
        act.sa_handler = exception_signal_handle;
        sigemptyset(&act.sa_mask);
        act.sa_flags = 0;
        sigaction(SIGBUS, &act, 0);
        sigaction(SIGSEGV, &act, 0);

        expecting_exception_signal = true;
        MB();
    }
    print_dbg("%s: trying to read buf_ptr[0]\n", myname);
    int data_from_buf_ptr = buf_ptr[0];
    print_dbg("%s: read buf_ptr[0] get %d\n", myname, data_from_buf_ptr);
    if (pid == 0) {
        MB();
        expecting_exception_signal = false;
        MB();
        print_dbg("%s: should not be able to read buf_ptr[0] anymore!! aborting!!\n", myname);
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        print_dbg("%s: signaling child\n", myname);
        int msg = 1;
        ASSERT_EQ(write(write_fd, &msg, sizeof(int)), sizeof(int));
        print_dbg("%s: waiting for child to exit\n", myname);
        // Child should exit because of sigbus
        int child_exit_status = -EINVAL;
        ASSERT(wait(&child_exit_status) == pid);
        ASSERT_EQ(child_exit_status, EXIT_SUCCESS);
        print_dbg("%s: trying to read buf_ptr[0] after child exits\n", myname);
        data_from_buf_ptr = buf_ptr[0];
        print_dbg("%s: read buf_ptr[0] after child exits get %d\n", myname, data_from_buf_ptr);
        ASSERT_EQ(data_from_buf_ptr, mynumber);
        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
        ASSERTDRV(gfree_fn(&mhandle));
        ASSERT_EQ(gdr_close(g), 0);
    }

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_fork_after_gdr_map_cumemalloc)
{
    invalidation_fork_after_gdr_map<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_fork_after_gdr_map_vmmalloc)
{
    invalidation_fork_after_gdr_map<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * This unit test ensures that child cannot do gdr_map on what parent has
 * prepared with gdr_pin_buffer. This situation emulates when the parent
 * forgets that it has gdr_pin_buffer without gdr_map before doing fork.
 *
 * Step:
 * 1. Initilize CUDA and gdrcopy
 * 2. Do gdr_pin_buffer
 * 3. Fork the process
 *
 * 4.P Parent: Waiting for child to exit
 * 
 * 4.C Child: Attempt to do gdr_map on the parent's pinned buffer. gdrdrv is
 *     expected to prevent this case so that the child process cannot spy on
 *     the parent's GPU data.
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_fork_child_gdr_map_parent()
{
    expecting_exception_signal = false;
    MB();

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    ASSERT(pid >= 0);

    myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    if (pid == 0) {
        void *bar_ptr  = NULL;
        print_dbg("%s: attempting to gdr_map parent's pinned GPU memory\n", myname);
        ASSERT_NEQ(gdr_map(g, mh, &bar_ptr, size), 0);
        print_dbg("%s: cannot do gdr_map as expected\n", myname);
    }
    else {
        int child_exit_status = -EINVAL;
        ASSERT(wait(&child_exit_status) == pid);
        ASSERT_EQ(child_exit_status, EXIT_SUCCESS);

        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
        ASSERTDRV(gfree_fn(&mhandle));
        ASSERT_EQ(gdr_close(g), 0);

        finalize_cuda(0);
    }
}

BEGIN_GDRCOPY_TEST(invalidation_fork_child_gdr_map_parent_cumemalloc)
{
    invalidation_fork_child_gdr_map_parent<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_fork_child_gdr_map_parent_vmmalloc)
{
    invalidation_fork_child_gdr_map_parent<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * This unit test verifies that gpuMemFree of one process will not
 * unintentionally invalidate mapping on other processes.
 *
 * Step:
 * 1. Fork
 *
 * 2.P Parent: Init CUDA and gdrcopy, and do gdr_map.
 * 3.P Parent: Wait for child's signal.
 *
 * 2.C Child: Init CUDA and gdrcopy, and do gdr_map.
 * 3.C Child: Do gpuMemFree. This should unmap the gdr_map'ed region.
 * 4.C Child: Signal parent.
 *
 * 4.P Parent: Verify that it can still access its gdr_map'ed region. If gdrdrv
 *     does not implement correctly, it might invalidate parent's mapping as
 *     well.
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_fork_map_and_free()
{
    expecting_exception_signal = false;
    MB();

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ASSERT_NEQ(pipe(filedes_0), -1);
    ASSERT_NEQ(pipe(filedes_1), -1);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    ASSERT(pid >= 0);

    myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    if (pid == 0) {
        close(filedes_0[0]);
        close(filedes_1[1]);

        read_fd = filedes_1[0];
        write_fd = filedes_0[1];

        srand(rand());
    }
    else {
        close(filedes_0[1]);
        close(filedes_1[0]);

        read_fd = filedes_0[0];
        write_fd = filedes_1[1];
    }

    int mydata = (rand() % 1000) + 1;

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    gdr_t g = gdr_open_safe();

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = d_ptr - info.va;

    volatile int *buf_ptr = (volatile int *)((char *)bar_ptr + off);

    print_dbg("%s: writing buf_ptr[0] with %d\n", myname, mydata);
    buf_ptr[0] = mydata;

    if (pid == 0) {
        print_dbg("%s: calling gpuMemFree\n", myname);
        ASSERTDRV(gfree_fn(&mhandle));

        print_dbg("%s: signal parent that I have called gpuMemFree\n", myname);
        int msg = 1;
        ASSERT_EQ(write(write_fd, &msg, sizeof(int)), sizeof(int));
    }
    else {
        int cont = 0;
        do {
            print_dbg("%s: waiting for signal from child\n", myname);
            ASSERT_EQ(read(read_fd, &cont, sizeof(int)), sizeof(int));
            print_dbg("%s: received cont signal %d from child\n", myname, cont);
        } while (cont == 0);

        print_dbg("%s: trying to read buf_ptr[0]\n", myname);
        int data_from_buf_ptr = buf_ptr[0];
        print_dbg("%s: read buf_ptr[0] get %d\n", myname, data_from_buf_ptr);
        ASSERT_EQ(data_from_buf_ptr, mydata);
    }

    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

    if (pid > 0)
        ASSERTDRV(gfree_fn(&mhandle));

    ASSERT_EQ(gdr_close(g), 0);

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_fork_map_and_free_cumemalloc)
{
    invalidation_fork_map_and_free<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_fork_map_and_free_vmmalloc)
{
    invalidation_fork_map_and_free<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * Process A can intentionally share fd with Process B through unix socket.
 * This method may lead to sharing mappings of gdrcopy. Since CUDA contexts are
 * not sharable between processes, gdrcopy is also expected to be unsharable.
 * This unit test verifies that gdr_open's fd shared from another process is
 * not usable.
 *
 * Step:
 * 1. Fork
 *
 * 2.P Parent: Init CUDA and gdrcopy.
 * 3.P Parent: Share gdr_open's fd to child through unix socket.
 *
 * 2.C Child: Init CUDA.
 * 3.C Child: Receive the fd from parent.
 * 4.C Child: Attempt to do gdr_pin_buffer using this fd. gdrdrv should not
 *     allow it.
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_unix_sock_shared_fd_gdr_pin_buffer()
{
    expecting_exception_signal = false;
    MB();

    pid_t pid;
    int pair[2];
    int fd = -1;

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    ASSERT_EQ(socketpair(PF_UNIX, SOCK_DGRAM, 0, pair), 0);

    fflush(stdout);
    fflush(stderr);

    pid = fork();
    ASSERT(pid >= 0);
    const char *myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    CUdeviceptr d_ptr = d_A;

    if (pid == 0) {
        close(pair[1]);

        print_dbg("%s: Receiving fd from parent via unix socket\n", myname);
        fd = recvfd(pair[0]);
        ASSERT(fd >= 0);

        print_dbg("%s: Got fd %d\n", myname, fd);

        print_dbg("%s: Converting fd to gdr_t\n", myname);
        struct gdr _g;
        _g.fd = fd;
        gdr_t g = &_g;

        print_dbg("%s: Trying to do gdr_pin_buffer with the received fd\n", myname);
        gdr_mh_t mh;
        ASSERT_NEQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
        print_dbg("%s: Cannot do gdr_pin_buffer with the received fd as expected\n", myname);
    }
    else {
        close(pair[0]);

        print_dbg("%s: Calling gdr_open\n", myname);
        gdr_t g = gdr_open_safe();

        fd = g->fd;
        print_dbg("%s: Extracted fd from gdr_t got fd %d\n", myname, fd);

        print_dbg("%s: Sending fd to child via unix socket\n", myname);
        ASSERT(sendfd(pair[1], fd) >= 0);

        print_dbg("%s: Waiting for child to finish\n", myname);
        int child_exit_status = -EINVAL;
        ASSERT(wait(&child_exit_status) == pid);
        ASSERT_EQ(child_exit_status, EXIT_SUCCESS);
    }

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_unix_sock_shared_fd_gdr_pin_buffer_cumemalloc)
{
    invalidation_unix_sock_shared_fd_gdr_pin_buffer<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_unix_sock_shared_fd_gdr_pin_buffer_vmmalloc)
{
    invalidation_unix_sock_shared_fd_gdr_pin_buffer<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * Process A can intentionally share fd with Process B through unix socket.
 * This method may lead to sharing mappings of gdrcopy. Since CUDA contexts are
 * not sharable between processes, gdrcopy is also expected to be unsharable.
 * This unit test verifies that gdr_open's fd shared from another process is
 * not usable.
 *
 * Step:
 * 1. Fork
 *
 * 2.P Parent: Init CUDA and gdrcopy, and do gdr_pin_buffer
 * 3.P Parent: Share gdr_open's fd to child through unix socket.
 * 4.P Parent: Also share the handle returned from gdr_pin_buffer with child.
 *
 * 2.C Child: Init CUDA.
 * 3.C Child: Receive the fd and handle from parent.
 * 4.C Child: Attempt to do gdr_map using this fd and handle. gdrdrv should not
 *     allow it.
 */
template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void invalidation_unix_sock_shared_fd_gdr_map()
{
    expecting_exception_signal = false;
    MB();

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ASSERT_NEQ(pipe(filedes_0), -1);
    ASSERT_NEQ(pipe(filedes_1), -1);

    pid_t pid;
    int pair[2];
    int fd = -1;

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    ASSERT_EQ(socketpair(PF_UNIX, SOCK_DGRAM, 0, pair), 0);

    fflush(stdout);
    fflush(stderr);

    pid = fork();
    ASSERT(pid >= 0);
    const char *myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);
    if (pid == 0) {
        close(filedes_0[0]);
        close(filedes_1[1]);

        read_fd = filedes_1[0];
        write_fd = filedes_0[1];

        srand(rand());
    }
    else {
        close(filedes_0[1]);
        close(filedes_1[0]);

        read_fd = filedes_0[0];
        write_fd = filedes_1[1];
    }

    init_cuda(0);
    filter_fn();

    CUdeviceptr d_A;
    gpu_mem_handle_t mhandle;
    ASSERTDRV(galloc_fn(&mhandle, size, true, true));
    d_A = mhandle.ptr;

    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));
    ASSERTDRV(cuCtxSynchronize());

    CUdeviceptr d_ptr = d_A;

    if (pid == 0) {
        close(pair[1]);

        print_dbg("%s: Receiving fd from parent via unix socket\n", myname);
        fd = recvfd(pair[0]);
        ASSERT(fd >= 0);

        print_dbg("%s: Got fd %d\n", myname, fd);

        print_dbg("%s: Converting fd to gdr_t\n", myname);
        struct gdr _g;
        _g.fd = fd;
        gdr_t g = &_g;

        print_dbg("%s: Receiving gdr_memh_t from parent\n", myname);
        gdr_memh_t memh;
        ASSERT_EQ(read(read_fd, &memh, sizeof(gdr_memh_t)), sizeof(gdr_memh_t));
        print_dbg("%s: Got handle 0x%lx\n", myname, memh.handle);

        print_dbg("%s: Converting gdr_memh_t to gdr_mh_t\n", myname);
        gdr_mh_t mh;
        mh.h = (unsigned long)(&memh);

        print_dbg("%s: Attempting gdr_map\n", myname);
        void *bar_ptr  = NULL;
        ASSERT_NEQ(gdr_map(g, mh, &bar_ptr, size), 0);
        print_dbg("%s: Cannot do gdr_map as expected\n", myname);
    }
    else {
        close(pair[0]);

        print_dbg("%s: Calling gdr_open\n", myname);
        gdr_t g = gdr_open_safe();

        print_dbg("%s: Calling gdr_pin_buffer\n", myname);
        gdr_mh_t mh;
        ASSERT_EQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        fd = g->fd;
        print_dbg("%s: Extracted fd from gdr_t got fd %d\n", myname, fd);

        print_dbg("%s: Sending fd to child via unix socket\n", myname);
        ASSERT(sendfd(pair[1], fd) >= 0);

        gdr_memh_t *memh = (gdr_memh_t *)mh.h;
        print_dbg("%s: Extracted gdr_memh_t from gdr_mh_t got handle 0x%lx\n", myname, memh->handle);

        print_dbg("%s: Sending gdr_memh_t to child\n", myname);
        ASSERT_EQ(write(write_fd, memh, sizeof(gdr_memh_t)), sizeof(gdr_memh_t));

        print_dbg("%s: Waiting for child to finish\n", myname);
        int child_exit_status = -EINVAL;
        ASSERT(wait(&child_exit_status) == pid);
        ASSERT_EQ(child_exit_status, EXIT_SUCCESS);
    }

    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(invalidation_unix_sock_shared_fd_gdr_map_cumemalloc)
{
    invalidation_unix_sock_shared_fd_gdr_map<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(invalidation_unix_sock_shared_fd_gdr_map_vmmalloc)
{
    invalidation_unix_sock_shared_fd_gdr_map<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST

/**
 * Although the use of P2P tokens has been marked as depricated, CUDA still
 * supports it.  This unit test ensures that Process A cannot access GPU memory
 * of Process B by using tokens, which can be bruteforcedly generated.
 *
 * Step:
 * 1. Fork the process
 *
 * 2.P Parent: Allocate GPU memory and get tokens.
 * 3.P Parent: Send the cuMemAlloc'd ptr and the tokens to Child.
 * 4.P Parent: Waiting for Child to exit.
 *
 * 2.C Child: Waiting for ptr and tokens from Parent
 * 3.C Child: Attempt gdr_pin_buffer with the ptr and tokens. We expect that
 *     gdr_pin_buffer would fail
 */
BEGIN_GDRCOPY_TEST(invalidation_fork_child_gdr_pin_parent_with_tokens)
{
    expecting_exception_signal = false;
    MB();

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ASSERT_NEQ(pipe(filedes_0), -1);
    ASSERT_NEQ(pipe(filedes_1), -1);

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    fflush(stdout);
    fflush(stderr);

    CUdeviceptr d_A;
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens = {0,0};

    pid_t pid = fork();
    ASSERT(pid >= 0);

    myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    if (pid == 0) {
        close(filedes_0[0]);
        close(filedes_1[1]);

        read_fd = filedes_1[0];
        write_fd = filedes_0[1];

        gdr_t g = gdr_open_safe();

        ASSERT_EQ(read(read_fd, &d_A, sizeof(CUdeviceptr)), sizeof(CUdeviceptr));
        ASSERT_EQ(read(read_fd, &tokens, sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS)), sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS));

        print_dbg("%s: Received from parent tokens.p2pToken %llu, tokens.vaSpaceToken %u\n", myname, tokens.p2pToken, tokens.vaSpaceToken);

        gdr_mh_t mh;

        CUdeviceptr d_ptr = d_A;

        ASSERT_NEQ(gdr_pin_buffer(g, d_ptr, size, tokens.p2pToken, tokens.vaSpaceToken, &mh), 0);
    }
    else {
        close(filedes_0[1]);
        close(filedes_1[0]);

        read_fd = filedes_0[0];
        write_fd = filedes_1[1];

        init_cuda(0);

        gpu_mem_handle_t mhandle;
        ASSERTDRV(gpu_mem_alloc(&mhandle, size, true, true));
        d_A = mhandle.ptr;

        ASSERTDRV(cuPointerGetAttribute(&tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, d_A));

        print_dbg("%s: CUDA generated tokens.p2pToken %llu, tokens.vaSpaceToken %u\n", myname, tokens.p2pToken, tokens.vaSpaceToken);

        ASSERT_EQ(write(write_fd, &d_A, sizeof(CUdeviceptr)), sizeof(CUdeviceptr));
        ASSERT_EQ(write(write_fd, &tokens, sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS)), sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS));

        int child_exit_status = -EINVAL;
        ASSERT(wait(&child_exit_status) == pid);
        ASSERT_EQ(child_exit_status, EXIT_SUCCESS);

        ASSERTDRV(gpu_mem_free(&mhandle));

        finalize_cuda(0);
    }
}
END_GDRCOPY_TEST


struct mt_test_info {
    gpu_mem_handle_t mhandle;
    CUdeviceptr d_buf;
    void *mapped_d_buf;
    size_t size;
    gdr_t g;
    gdr_mh_t mh;
    bool use_barrier;
    pthread_barrier_t barrier;
    gpu_memfree_fn_t gfree_fn;
};

void *thr_fun_setup(void *data)
{
    mt_test_info *pt = static_cast<mt_test_info*>(data);
    ASSERT(pt);
    print_dbg("pinning\n");
    ASSERT_EQ(gdr_pin_buffer(pt->g, pt->d_buf, pt->size, 0, 0, &pt->mh), 0);
    ASSERT_NEQ(pt->mh, null_mh);
    print_dbg("mapping\n");
    ASSERT_EQ(gdr_map(pt->g, pt->mh, &pt->mapped_d_buf, pt->size), 0);
    if (pt->use_barrier)
        pthread_barrier_wait(&pt->barrier);
    return NULL;
}

void *thr_fun_teardown(void *data)
{
    mt_test_info *pt = static_cast<mt_test_info*>(data);
    ASSERT(pt);
    if (pt->use_barrier)
        pthread_barrier_wait(&pt->barrier);
    print_dbg("unmapping\n");
    ASSERT_EQ(gdr_unmap(pt->g, pt->mh, pt->mapped_d_buf, pt->size), 0);
    pt->mapped_d_buf = 0;
    print_dbg("unpinning\n");
    ASSERT_EQ(gdr_unpin_buffer(pt->g, pt->mh), 0);
    pt->mh = null_mh;
    return NULL;
}

void *thr_fun_combined(void *data)
{
    mt_test_info *pt = static_cast<mt_test_info*>(data);
    ASSERT(pt);
    ASSERT(!pt->use_barrier);
    thr_fun_setup(data);
    thr_fun_teardown(data);
    return NULL;
}

void *thr_fun_cleanup(void *data)
{
    mt_test_info *pt = static_cast<mt_test_info*>(data);
    ASSERT(pt);
    ASSERT_EQ(gdr_close(pt->g), 0);
    pt->g = 0;
    ASSERTDRV(pt->gfree_fn(&pt->mhandle));
    pt->d_buf = 0;
    return NULL;
}

template <gpu_memalloc_fn_t galloc_fn, gpu_memfree_fn_t gfree_fn, filter_fn_t filter_fn>
void basic_child_thread_pins_buffer()
{
    const size_t _size = GPU_PAGE_SIZE * 16;
    mt_test_info t;
    memset(&t, 0, sizeof(mt_test_info));
    t.size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    init_cuda(0);
    filter_fn();

    t.gfree_fn = gfree_fn;

    ASSERTDRV(galloc_fn(&t.mhandle, t.size, true, true));
    t.d_buf = t.mhandle.ptr;

    ASSERTDRV(cuMemsetD8(t.d_buf, 0xA5, t.size));
    ASSERTDRV(cuCtxSynchronize());

    t.g = gdr_open_safe();
    {
        pthread_t tid;
        t.use_barrier = false;
        print_dbg("spawning single child thread\n");
        ASSERT_EQ(pthread_create(&tid, NULL, thr_fun_combined, &t), 0);
        ASSERT_EQ(pthread_join(tid, NULL), 0);
    }
    {
        pthread_t tid[2];
        ASSERT_EQ(pthread_barrier_init(&t.barrier, NULL, 2), 0);
        t.use_barrier = true;
        print_dbg("spawning two children threads, splitting setup and teardown\n");
        ASSERT_EQ(pthread_create(&tid[0], NULL, thr_fun_setup, &t), 0);
        ASSERT_EQ(pthread_create(&tid[1], NULL, thr_fun_teardown, &t), 0);
        ASSERT_EQ(pthread_join(tid[0], NULL), 0);
        ASSERT_EQ(pthread_join(tid[1], NULL), 0);
    }
    {
        pthread_t tid[2];
        t.use_barrier = false;
        mt_test_info t2 = t;
        print_dbg("spawning two children threads, concurrently pinning and mapping the same buffer\n");
        ASSERT_EQ(pthread_create(&tid[0], NULL, thr_fun_combined, &t), 0);
        ASSERT_EQ(pthread_create(&tid[1], NULL, thr_fun_combined, &t2), 0);
        ASSERT_EQ(pthread_join(tid[0], NULL), 0);
        ASSERT_EQ(pthread_join(tid[1], NULL), 0);
    }
    {
        pthread_t tid;
        print_dbg("spawning cleanup child thread\n");
        ASSERT_EQ(pthread_create(&tid, NULL, thr_fun_cleanup, &t), 0);
        ASSERT_EQ(pthread_join(tid, NULL), 0);
    }
    finalize_cuda(0);
}

BEGIN_GDRCOPY_TEST(basic_child_thread_pins_buffer_cumemalloc)
{
    basic_child_thread_pins_buffer<gpu_mem_alloc, gpu_mem_free, null_filter>();
}
END_GDRCOPY_TEST

BEGIN_GDRCOPY_TEST(basic_child_thread_pins_buffer_vmmalloc)
{
    basic_child_thread_pins_buffer<gpu_vmm_alloc, gpu_vmm_free, vmm_filter>();
}
END_GDRCOPY_TEST


int main(int argc, char *argv[])
{
    int c;

    while ((c = getopt(argc, argv, "vh")) != -1) {
        switch (c) {
            case 'v':
                gdrcopy::test::print_dbg_msg = true;
                break;
            case 'h':
                cout << "Usage: " << argv[0] << " [-v] [-h]" << endl;
                return EXIT_SUCCESS;
            case '?':
                if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return EXIT_FAILURE;
            default:
                abort();
        }
    }

    Suite *s = suite_create("Sanity");

    TCase *tc_basic = tcase_create("Basic");
    TCase *tc_data_validation = tcase_create("Data Validation");
    TCase *tc_invalidation = tcase_create("Invalidation");

    SRunner *sr = srunner_create(s);

    int nf;

    suite_add_tcase(s, tc_basic);
    suite_add_tcase(s, tc_data_validation);
    suite_add_tcase(s, tc_invalidation);

    tcase_add_test(tc_basic, basic_cumemalloc);
    tcase_add_test(tc_basic, basic_with_tokens);
    tcase_add_test(tc_basic, basic_unaligned_mapping);
    tcase_add_test(tc_basic, basic_child_thread_pins_buffer_cumemalloc);

    tcase_add_test(tc_data_validation, data_validation_cumemalloc);

    tcase_add_test(tc_invalidation, invalidation_access_after_gdr_close_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_access_after_free_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_two_mappings_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_access_after_free_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_after_gdr_map_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_child_gdr_map_parent_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_map_and_free_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_unix_sock_shared_fd_gdr_pin_buffer_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_unix_sock_shared_fd_gdr_map_cumemalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_child_gdr_pin_parent_with_tokens);


    #if CUDA_VERSION >= 11000
    // VMM with GDR support is available from CUDA 11.0
    tcase_add_test(tc_basic, basic_vmmalloc);
    tcase_add_test(tc_basic, basic_child_thread_pins_buffer_vmmalloc);

    tcase_add_test(tc_data_validation, data_validation_vmmalloc);

    tcase_add_test(tc_invalidation, invalidation_access_after_gdr_close_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_access_after_free_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_two_mappings_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_access_after_free_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_after_gdr_map_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_child_gdr_map_parent_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_fork_map_and_free_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_unix_sock_shared_fd_gdr_pin_buffer_vmmalloc);
    tcase_add_test(tc_invalidation, invalidation_unix_sock_shared_fd_gdr_map_vmmalloc);
    #endif

    tcase_set_timeout(tc_basic, 60);
    tcase_set_timeout(tc_data_validation, 60);
    tcase_set_timeout(tc_invalidation, 180);

    srunner_run_all(sr, CK_ENV);
    nf = srunner_ntests_failed(sr);
    srunner_free(sr);

    return nf == 0 ? 0 : 1;
}


/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
