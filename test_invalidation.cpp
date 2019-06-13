/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <stdarg.h>
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

using namespace std;

#include "gdrapi.h"
#include "common.hpp"

static bool _print_dbg_msg = false;

static void print_dbg(const char* fmt, ...)
{
    if (_print_dbg_msg) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
    }
}


volatile bool expecting_exception_signal = false;

void exception_signal_handle(int sig)
{
    if (expecting_exception_signal) {
        print_dbg("Get signal %d as expected\n", sig);
        exit(EXIT_SUCCESS);
    }
    ck_abort_msg("Unexpectedly get exception signal");
}

/**
 * Sends given file descriptior via given socket
 *
 * @param socket to be used for fd sending
 * @param fd to be sent
 * @return sendmsg result
 *
 * @note socket should be (PF_UNIX, SOCK_DGRAM)
 */
int sendfd(int socket, int fd) {
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
int recvfd(int socket) {
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

struct gdr {
    int fd;
};

typedef struct { 
    uint32_t handle;
    unsigned mapped:1;
    unsigned wc_mapping:1;
} gdr_memh_t;

/**
 * This unit test ensures that accessing to gdr_map'ed region is not possible
 * after cuMemFree.
 *
 * Step:
 * 1. Initialize CUDA and gdrcopy
 * 2. Do gdr_map(..., &bar_ptr, ...)
 * 3. Do cuMemFree
 * 4. Attempt to access to bar_ptr after 3. should fail
 */
START_TEST(invalidation_access_after_cumemfree)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_access_after_cumemfree\n");

    struct sigaction act;
    act.sa_handler = exception_signal_handle;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGBUS, &act, 0);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    int mydata = (rand() % 1000) + 1;

    void *dummy;
    // Let libcudart initialize CUDA for us
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;
    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
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

    print_dbg("Calling cuMemFree\n");
    ASSERTDRV(cuMemFree(d_A));
    
    print_dbg("Trying to read buf_ptr[0] after cuMemFree\n");
    expecting_exception_signal = true;
    int data_from_buf_ptr = buf_ptr[0];
    expecting_exception_signal = false;

    ck_assert_msg(data_from_buf_ptr != mydata, "Got the same data after cuMemFree!!");
    
    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    ASSERT_EQ(gdr_close(g), 0);

    print_dbg("End invalidation_access_after_cumemfree\n");
}
END_TEST

/**
 * This unit test ensures that cuMemFree destroys only the mapping it is
 * corresponding to.
 *
 * Step:
 * 1. Initialize CUDA and gdrcopy
 * 2. cuMemAlloc(&d_A, ...); cuMemAlloc(&d_B, ...)
 * 3. Do gdr_map(..., &bar_ptr_A, ...) of d_A
 * 4. Do gdr_map(..., &bar_ptr_B, ...) of d_B
 * 5. Do cuMemFree(d_A)
 * 6. Verify that bar_ptr_B is still accessible 
 */
START_TEST(invalidation_two_mappings)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_two_mappings\n");

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    int mydata = (rand() % 1000) + 1;

    void *dummy;
    // Let libcudart initialize CUDA for us
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A[2];

    for (int i = 0; i < 2; ++i) {
        ASSERTDRV(cuMemAlloc(&d_A[i], size));
        ASSERTDRV(cuMemsetD8(d_A[i], 0x95, size));

        unsigned int flag = 1;
        ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A[i]));
    }

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh[2];

    volatile int *buf_ptr[2];
    void *bar_ptr[2];

    print_dbg("Mapping bar1\n");
    for (int i = 0; i < 2; ++i) {
        CUdeviceptr d_ptr = d_A[i];

        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
        ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh[i]), 0);
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
    ck_assert_int_eq(buf_ptr[0][0], mydata);
    ck_assert_int_eq(buf_ptr[1][0], mydata + 1);

    print_dbg("cuMemFree and thus destroying the first mapping\n");
    ASSERTDRV(cuMemFree(d_A[0]));

    print_dbg("Trying to read and validate the data from the second mapping after the first mapping has been destroyed\n");
    ck_assert_int_eq(buf_ptr[1][0], mydata + 1);

    ASSERTDRV(cuMemFree(d_A[1]));
    
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(gdr_unmap(g, mh[i], bar_ptr[i], size), 0);
        ASSERT_EQ(gdr_unpin_buffer(g, mh[i]), 0);
    }

    ASSERT_EQ(gdr_close(g), 0);

    print_dbg("End invalidation_two_mappings\n");
}
END_TEST

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
 * 3.P Parent: Do gdr_map then cuMemFree without gdr_unmap
 * 4.P Parent: Signal child and wait for child's signal
 *
 * 3.C Child: Initialize CUDA and gdrcopy
 * 4.C Child: Do gdr_map, signal parent, and wait for parent's signal
 *
 * 5.P Parent: Check whether it can access to its gdr_map'ed data or not and
 *     compare with the data written by child. If gdrdrv does not handle
 *     invalidation properly, child's data will be leaked to parent.
 */
START_TEST(invalidation_fork_access_after_cumemfree)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_fork_access_after_cumemfree\n");

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ck_assert_int_ne(pipe(filedes_0), -1);
    ck_assert_int_ne(pipe(filedes_1), -1);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    pid_t pid = fork();
    ck_assert_int_ge(pid, 0);

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
            ck_assert_int_eq(read(read_fd, &cont, sizeof(int)), sizeof(int));
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

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
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
        ck_assert_int_eq(write(write_fd, &mydata, sizeof(int)), sizeof(int));

        int cont = 0;
        print_dbg("%s: waiting for signal from parent before calling cuMemFree\n", myname);
        do {
            ck_assert_int_eq(read(read_fd, &cont, sizeof(int)), sizeof(int));
        } while (cont != 1);
    }

    print_dbg("%s: read buf_ptr[0] before cuMemFree get %d\n", myname, buf_ptr[0]);

    print_dbg("%s: calling cuMemFree\n", myname);
    ASSERTDRV(cuMemFree(d_A));

    if (pid > 0) {
        int msg = 1;
        ck_assert_int_eq(write(write_fd, &msg, sizeof(int)), sizeof(int));
        int child_data = 0;
        print_dbg("%s: waiting for child write signal\n", myname);
        do {
            ck_assert_int_eq(read(read_fd, &child_data, sizeof(int)), sizeof(int));
        } while (child_data == 0);

        print_dbg("%s: trying to read buf_ptr[0]\n", myname);
        expecting_exception_signal = true;
        int data_from_buf_ptr = buf_ptr[0];
        expecting_exception_signal = false;

        print_dbg("%s: read buf_ptr[0] after child write get %d\n", myname, data_from_buf_ptr);
        print_dbg("%s: child data is %d\n", myname, child_data);
        ck_assert_int_eq(write(write_fd, &msg, sizeof(int)), sizeof(int));
        ck_assert_msg(child_data != data_from_buf_ptr, "Data from the child process should not be visible on the parent process via bar mapping!!! Security breached!!!");
    }

    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

    ASSERT_EQ(gdr_close(g), 0);

    print_dbg("End invalidation_fork_access_after_cumemfree\n");
}
END_TEST

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
START_TEST(invalidation_fork_after_gdr_map)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_fork_after_gdr_map\n");

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ck_assert_int_ne(pipe(filedes_0), -1);
    ck_assert_int_ne(pipe(filedes_1), -1);

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    void *bar_ptr  = NULL;
    ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);

    gdr_info_t info;
    ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
    int off = d_ptr - info.va;

    volatile int *buf_ptr = (volatile int *)((char *)bar_ptr + off);

    pid_t pid = fork();
    ck_assert_int_ge(pid, 0);

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
            ck_assert_int_eq(read(read_fd, &cont, sizeof(int)), sizeof(int));
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
    }
    print_dbg("%s: trying to read buf_ptr[0]\n", myname);
    int data_from_buf_ptr = buf_ptr[0];
    print_dbg("%s: read buf_ptr[0] get %d\n", myname, data_from_buf_ptr);
    if (pid == 0) {
        expecting_exception_signal = false;
        print_dbg("%s: should not be able to read buf_ptr[0] anymore!! aborting!!\n", myname);
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        print_dbg("%s: signaling child\n", myname);
        int msg = 1;
        ck_assert_int_eq(write(write_fd, &msg, sizeof(int)), sizeof(int));
        print_dbg("%s: waiting for child to exit\n", myname);
        // Child should exit because of sigbus
        int child_exit_status = -EINVAL;
        ck_assert_int_eq(wait(&child_exit_status), pid);
        ck_assert_int_eq(child_exit_status, EXIT_SUCCESS);
        print_dbg("%s: trying to read buf_ptr[0] after child exits\n", myname);
        data_from_buf_ptr = buf_ptr[0];
        print_dbg("%s: read buf_ptr[0] after child exits get %d\n", myname, data_from_buf_ptr);
        ck_assert_int_eq(data_from_buf_ptr, mynumber);
        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
        ASSERTDRV(cuMemFree(d_A));
        ASSERT_EQ(gdr_close(g), 0);
    }

    print_dbg("End invalidation_fork_after_gdr_map\n");
}
END_TEST

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
START_TEST(invalidation_fork_child_gdr_map_parent)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_fork_child_gdr_map_parent\n");

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
    ASSERT_NEQ(mh, null_mh);

    pid_t pid = fork();
    ck_assert_int_ge(pid, 0);

    myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    if (pid == 0) {
        void *bar_ptr  = NULL;
        print_dbg("%s: attempting to gdr_map parent's pinned GPU memory\n", myname);
        ck_assert_int_ne(gdr_map(g, mh, &bar_ptr, size), 0);
        print_dbg("%s: cannot do gdr_map as expected\n", myname);
    }
    else {
        int child_exit_status = -EINVAL;
        ck_assert_int_eq(wait(&child_exit_status), pid);
        ck_assert_int_eq(child_exit_status, EXIT_SUCCESS);

        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
        ASSERTDRV(cuMemFree(d_A));
        ASSERT_EQ(gdr_close(g), 0);
    }
    print_dbg("End invalidation_fork_child_gdr_map_parent\n");
}
END_TEST

/**
 * This unit test verifies that cuMemFree of one process will not
 * unintentionally invalidate mapping on other processes.
 *
 * Step:
 * 1. Fork
 *
 * 2.P Parent: Init CUDA and gdrcopy, and do gdr_map.
 * 3.P Parent: Wait for child's signal.
 *
 * 2.C Child: Init CUDA and gdrcopy, and do gdr_map.
 * 3.C Child: Do cuMemFree. This should unmap the gdr_map'ed region.
 * 4.C Child: Signal parent.
 *
 * 4.P Parent: Verify that it can still access its gdr_map'ed region. If gdrdrv
 *     does not implement correctly, it might invalidate parent's mapping as
 *     well.
 */
START_TEST(invalidation_fork_map_and_free)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_fork_map_and_free\n");

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ck_assert_int_ne(pipe(filedes_0), -1);
    ck_assert_int_ne(pipe(filedes_1), -1);

    srand(time(NULL));

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    const char *myname;

    pid_t pid = fork();
    ck_assert_int_ge(pid, 0);

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

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;

    CUdeviceptr d_ptr = d_A;

    // tokens are optional in CUDA 6.0
    // wave out the test if GPUDirectRDMA is not enabled
    ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
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
        print_dbg("%s: calling cuMemFree\n", myname);
        ASSERTDRV(cuMemFree(d_A));

        print_dbg("%s: signal parent that I have called cuMemFree\n", myname);
        int msg = 1;
        ck_assert_int_eq(write(write_fd, &msg, sizeof(int)), sizeof(int));
    }
    else {
        int cont = 0;
        do {
            print_dbg("%s: waiting for signal from child\n", myname);
            ck_assert_int_eq(read(read_fd, &cont, sizeof(int)), sizeof(int));
            print_dbg("%s: received cont signal %d from child\n", myname, cont);
        } while (cont == 0);

        print_dbg("%s: trying to read buf_ptr[0]\n", myname);
        int data_from_buf_ptr = buf_ptr[0];
        print_dbg("%s: read buf_ptr[0] get %d\n", myname, data_from_buf_ptr);
        ck_assert_int_eq(data_from_buf_ptr, mydata);
    }

    ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
    ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);

    if (pid > 0)
        ASSERTDRV(cuMemFree(d_A));

    ASSERT_EQ(gdr_close(g), 0);

    print_dbg("End invalidation_fork_map_and_free\n");
}
END_TEST

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
START_TEST(invalidation_unix_sock_shared_fd_gdr_pin_buffer)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_unix_sock_shared_fd_gdr_pin_buffer\n");

    pid_t pid;
    int pair[2];
    int fd = -1;

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    ck_assert_int_eq(socketpair(PF_UNIX, SOCK_DGRAM, 0, pair), 0);

    pid = fork();
    ck_assert_int_ge(pid, 0);
    const char *myname = pid == 0 ? "child" : "parent";

    print_dbg("%s: Start\n", myname);

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    CUdeviceptr d_ptr = d_A;

    if (pid == 0) {
        close(pair[1]);

        print_dbg("%s: Receiving fd from parent via unix socket\n", myname);
        fd = recvfd(pair[0]);
        ck_assert_int_ge(fd, 0);

        print_dbg("%s: Got fd %d\n", myname, fd);

        print_dbg("%s: Converting fd to gdr_t\n", myname);
        struct gdr _g;
        _g.fd = fd;
        gdr_t g = &_g;

        print_dbg("%s: Trying to do gdr_pin_buffer with the received fd\n", myname);
        gdr_mh_t mh;
        ck_assert_int_ne(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
        print_dbg("%s: Cannot do gdr_pin_buffer with the received fd as expected\n", myname);
    }
    else {
        close(pair[0]);

        print_dbg("%s: Calling gdr_open\n", myname);
        gdr_t g = gdr_open();
        ASSERT_NEQ(g, (void*)0);

        fd = g->fd;
        print_dbg("%s: Extracted fd from gdr_t got fd %d\n", myname, fd);
        
        print_dbg("%s: Sending fd to child via unix socket\n", myname);
        ck_assert_int_ge(sendfd(pair[1], fd), 0);

        print_dbg("%s: Waiting for child to finish\n", myname);
        int child_exit_status = -EINVAL;
        ck_assert_int_eq(wait(&child_exit_status), pid);
        ck_assert_int_eq(child_exit_status, EXIT_SUCCESS);
    }

    print_dbg("End invalidation_unix_sock_shared_fd_gdr_pin_buffer\n");
}
END_TEST

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
START_TEST(invalidation_unix_sock_shared_fd_gdr_map)
{
    expecting_exception_signal = false;

    print_dbg("Start invalidation_unix_sock_shared_fd_gdr_map\n");

    int filedes_0[2];
    int filedes_1[2];
    int read_fd;
    int write_fd;
    ck_assert_int_ne(pipe(filedes_0), -1);
    ck_assert_int_ne(pipe(filedes_1), -1);

    pid_t pid;
    int pair[2];
    int fd = -1;

    const size_t _size = sizeof(int) * 16;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    ck_assert_int_eq(socketpair(PF_UNIX, SOCK_DGRAM, 0, pair), 0);

    pid = fork();
    ck_assert_int_ge(pid, 0);
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

    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    CUdeviceptr d_A;
    ASSERTDRV(cuMemAlloc(&d_A, size));
    ASSERTDRV(cuMemsetD8(d_A, 0x95, size));

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));

    CUdeviceptr d_ptr = d_A;

    if (pid == 0) {
        close(pair[1]);

        print_dbg("%s: Receiving fd from parent via unix socket\n", myname);
        fd = recvfd(pair[0]);
        ck_assert_int_ge(fd, 0);

        print_dbg("%s: Got fd %d\n", myname, fd);

        print_dbg("%s: Converting fd to gdr_t\n", myname);
        struct gdr _g;
        _g.fd = fd;
        gdr_t g = &_g;

        print_dbg("%s: Receiving gdr_memh_t from parent\n", myname);
        gdr_memh_t memh;
        ck_assert_int_eq(read(read_fd, &memh, sizeof(gdr_memh_t)), sizeof(gdr_memh_t));
        print_dbg("%s: Got handle 0x%lx\n", myname, memh.handle);

        print_dbg("%s: Converting gdr_memh_t to gdr_mh_t\n", myname);
        gdr_mh_t mh;
        mh.h = (unsigned long)(&memh);

        print_dbg("%s: Attempting gdr_map\n", myname);
        void *bar_ptr  = NULL;
        ck_assert_int_ne(gdr_map(g, mh, &bar_ptr, size), 0);
        print_dbg("%s: Cannot do gdr_map as expected\n", myname);
    }
    else {
        close(pair[0]);

        print_dbg("%s: Calling gdr_open\n", myname);
        gdr_t g = gdr_open();
        ASSERT_NEQ(g, (void*)0);

        print_dbg("%s: Calling gdr_pin_buffer\n", myname);
        gdr_mh_t mh;
        ck_assert_int_eq(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, null_mh);

        fd = g->fd;
        print_dbg("%s: Extracted fd from gdr_t got fd %d\n", myname, fd);

        print_dbg("%s: Sending fd to child via unix socket\n", myname);
        ck_assert_int_ge(sendfd(pair[1], fd), 0);

        gdr_memh_t *memh = (gdr_memh_t *)mh.h;
        print_dbg("%s: Extracted gdr_memh_t from gdr_mh_t got handle 0x%lx\n", myname, memh->handle);

        print_dbg("%s: Sending gdr_memh_t to child\n", myname);
        ck_assert_int_eq(write(write_fd, memh, sizeof(gdr_memh_t)), sizeof(gdr_memh_t));

        print_dbg("%s: Waiting for child to finish\n", myname);
        int child_exit_status = -EINVAL;
        ck_assert_int_eq(wait(&child_exit_status), pid);
        ck_assert_int_eq(child_exit_status, EXIT_SUCCESS);
    }

    print_dbg("End invalidation_unix_sock_shared_fd_gdr_map\n");
}
END_TEST


int main(int argc, char *argv[])
{
    int c;

    while ((c = getopt(argc, argv, "h::v::")) != -1) {
        switch (c) {
            case 'v':
                _print_dbg_msg = true;
                print_dbg("Enable debug message\n");
                break;
            case 'h':
                cout << "Usage: " << argv[0] << " [-v] [-h]" << endl;
                break;
            case '?':
                if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;
            default:
                abort();
        }
    }

    Suite *s = suite_create("Invalidation");
    TCase *tc = tcase_create("Invalidation");
    SRunner *sr = srunner_create(s);

    int nf;

    suite_add_tcase(s, tc);
    tcase_add_test(tc, invalidation_access_after_cumemfree);
    tcase_add_test(tc, invalidation_two_mappings);
    tcase_add_test(tc, invalidation_fork_access_after_cumemfree);
    tcase_add_test(tc, invalidation_fork_after_gdr_map);
    tcase_add_test(tc, invalidation_fork_child_gdr_map_parent);
    tcase_add_test(tc, invalidation_fork_map_and_free);
    tcase_add_test(tc, invalidation_unix_sock_shared_fd_gdr_pin_buffer);
    tcase_add_test(tc, invalidation_unix_sock_shared_fd_gdr_map);

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
