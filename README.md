# gdrcopy

A low-latency GPU memory copy library based on NVIDIA GPUDirect RDMA
technology.


## Introduction

While GPUDirect RDMA is meant for direct access to GPU memory from
third-party devices, it is possible to use these same APIs to create
perfectly valid CPU mappings of the GPU memory.

The advantage of a CPU driven copy is the essencially zero latency
involved in the copy process. This might be useful when low latencies
are required.


## Disclaimer

This is just for technology demonstration purposes. In particular this
is not an NVIDIA-supported product.

The library relies on a small kernel-mode driver (gdrdrv) which has
bug(s) and can even crash your machine.  In particular, there is a
latent bug related to the concurrent invalidation of mappings and
memory deallocation.


## What is inside

Basically, gdrcopy offers the infrastructure to create user-space
mappings of GPU memory, which can then be manipulated as if it was
plain host memory (caveats apply here).

A simple by-product of it is a copy library with the following characteristics:
- Zero-latency, as it is driven by the CPU.

- An initial memory "pinning" phase is required, which is potentially expensive,
  10us-1ms depending on the buffer size.

- Fast H-D, because of write-combining. H-D bandwidth is 6-8GB/s on Ivy
  Bridge Xeon but it is subject to NUMA effects.

- Slow D-H, because the GPU BAR, which backs the mappings, can't be
  prefetched and so burst reads transactions are not generated through
  PCIE

The library comes with two tests:
- validate, which is a simple application testing the APIs.
- copybw, a minimal application which calculates the R/W bandwidth.



## Requirements

GPUDirect RDMA requires an NVIDIA Tesla and Quadro class GPUs based on
Kepler/Maxwell, see [GPUDirect
RDMA](http://developer.nvidia.com/gpudirect). 

For more technical informations, please refer to the official
GPUDirect RDMA [design
document](http://docs.nvidia.com/cuda/gpudirect-rdma).

The device driver requires CUDA >= 5.0.
The library and tests require CUDA >= 6.0 and/or display driver >= 331.14.

Developed and tested on RH6.x. The only supported architecture is
Linux x86_64 so far.

root priviledges are necessary to load/install the kernel-mode device
driver.


## Build & execution

Build:
```shell
$ cd gdrcopy
$ make PREFIX=<install path dir> CUDA=<cuda install path> all install
```

Install kernel-mode driver (root/sudo caps required):
```shell
$ ./insmod.sh
```

Prepare environment:
```shell
$ export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

Execute provided tests:
```shell
$ ./validate
buffer size: 327680
check 1: direct access + read back via cuMemcpy D->H
check 2: gdr_copy_to_bar() + read back via cuMemcpy D->H
check 3: gdr_copy_to_bar() + read back via gdr_copy_from_bar()
check 4: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + extra_dwords=5
$ ./copybw
testing size: 4096
rounded size: 65536
device ptr: 5046c0000
bar_ptr: 0x7f8cff410000
info.va: 5046c0000
info.mapped_size: 65536
info.page_size: 65536
page offset: 0
user-space pointer:0x7f8cff410000
BAR writing test...
BAR1 write BW: 9549.25MB/s
BAR reading test...
BAR1 read BW: 1.50172MB/s
unmapping buffer
unpinning buffer
closing gdrdrv
```

## TODO

- add RPM specs for both library and kernel-mode driver
- explore use of DKMS for kernel-mode driver
- Conditionally use P2P tokens, to be compatible with CUDA 5.x.
- Implement an event-queue mechanism in gdrdrv to deliver mapping
  invalidation events to user-space applications.
