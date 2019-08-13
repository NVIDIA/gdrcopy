# gdrcopy

A low-latency GPU memory copy library based on NVIDIA GPUDirect RDMA
technology.


## Introduction

While GPUDirect RDMA is meant for direct access to GPU memory from
third-party devices, it is possible to use these same APIs to create
perfectly valid CPU mappings of the GPU memory.

The advantage of a CPU driven copy is the very small overhead
involved. That might be useful when low latencies are required.


## What is inside

GDRCopy offers the infrastructure to create user-space mappings of GPU memory,
which can then be manipulated as if it was plain host memory (caveats apply
here).

A simple by-product of it is a copy library with the following characteristics:
- very low overhead, as it is driven by the CPU. As a reference, currently a 
  cudaMemcpy can incur in a 6-7us overhead.

- An initial memory *pinning* phase is required, which is potentially expensive,
  10us-1ms depending on the buffer size.

- Fast H-D, because of write-combining. H-D bandwidth is 6-8GB/s on Ivy
  Bridge Xeon but it is subject to NUMA effects.

- Slow D-H, because the GPU BAR, which backs the mappings, can't be
  prefetched and so burst reads transactions are not generated through
  PCIE

The library comes with two tests:
- sanity, which contains unit tests for the library and the driver.
- copybw, a minimal application which calculates the R/W bandwidth.


## Restrictions

This library only works with regular CUDA device memory, as returned by
cudaMalloc. In particular, it does not work with CUDA managed memory.


## Requirements

GPUDirect RDMA requires an NVIDIA Tesla and Quadro class GPUs based on Kepler,
Pascal, Volta, or Turing, see [GPUDirect
RDMA](http://developer.nvidia.com/gpudirect).  For more technical informations,
please refer to the official GPUDirect RDMA [design
document](http://docs.nvidia.com/cuda/gpudirect-rdma).

The device driver requires GPU display driver >= 331.14. The library and tests
require CUDA >= 6.0. Additionally, the _sanity_ test requires check >= 0.9.8 and
subunit.

```shell
# On RHEL
$ sudo yum install check subunit subunit-devel

# On Debian
$ sudo apt install check libsubunit0 libsubunit-dev
```

Developed and tested on RH7.x and Ubuntu18_04. The supported architectures are
Linux x86_64 and ppc64le.

root privileges are necessary to load/install the kernel-mode device
driver.


## Build and installation

We provide three ways for building and installing GDRCopy

### rpm package

```shell
$ cd packages
$ CUDA=<cuda-install-top-dir> ./build-rpm-packages.sh
$ sudo rpm -Uvh gdrcopy-kmod-<version>.<platform>.rpm
$ sudo rpm -Uvh gdrcopy-<version>.<platform>.rpm
$ sudo rpm -Uvh gdrcopy-devel-<version>.<platform>.rpm
```

### deb package

```shell
$ cd packages
$ CUDA=<cuda-install-top-dir> ./build-deb-packages.sh
$ sudo dpkg -i gdrdrv-dkms_<version>_<platform>.deb
$ sudo dpkg -i gdrcopy_<version>_<platform>.deb
```

### from source

```shell
$ make PREFIX=<install-to-this-location> CUDA=<cuda-install-top-dir> all install
$ sudo ./insmod.sh
```


## Tests

Execute provided tests:
```shell
$ sanity
Running suite(s): Sanity
100%: Checks: 11, Failures: 0, Errors: 0


$ copybw
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

## NUMA effects

Depending on the platform architecture, like where the GPU are placed in
the PCIe topology, performance may suffer if the processor which is driving
the copy is not the one which is hosting the GPU, for example in a
multi-socket server.

In the example below, the K40m and K80 GPU are respectively hosted by
socket0 and socket1. By explicitly playing with the OS process and memory
affinity, it is possible to run the test onto the optimal processor:

```shell
$ GDRCOPY_ENABLE_LOGGING=1 GDRCOPY_LOG_LEVEL=0 LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH numactl -N 0 -l copybw -d 0 -s $((64 * 1024)) -o $((0 * 1024)) -c $((64 * 1024))
GPU id:0 name:Tesla K40m PCI domain: 0 bus: 2 device: 0
GPU id:1 name:Tesla K80 PCI domain: 0 bus: 132 device: 0
GPU id:2 name:Tesla K80 PCI domain: 0 bus: 133 device: 0
selecting device 0
testing size: 65536
rounded size: 65536
device ptr: 2305ba0000
bar_ptr: 0x7fe60956c000
info.va: 2305ba0000
info.mapped_size: 65536
info.page_size: 65536
page offset: 0
user-space pointer:0x7fe60956c000
BAR writing test, size=65536 offset=0 num_iters=10000
DBG:  sse4_1=1 avx=1 sse=1 sse2=1
DBG:  using AVX implementation of gdr_copy_to_bar
BAR1 write BW: 9793.23MB/s
BAR reading test, size=65536 offset=0 num_iters=100
DBG:  using SSE4_1 implementation of gdr_copy_from_bar
BAR1 read BW: 787.957MB/s
unmapping buffer
unpinning buffer
closing gdrdrv
```

or on the other one:
```shell
drossetti@drossetti-hsw0 16:52 (1181) gdrcopy>GDRCOPY_ENABLE_LOGGING=1 GDRCOPY_LOG_LEVEL=0 LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH numactl -N 1 -l copybw -d 0 -s $((64 * 1024)) -o $((0 * 1024)) -c $((64 * 1024))
GPU id:0 name:Tesla K40m PCI domain: 0 bus: 2 device: 0
GPU id:1 name:Tesla K80 PCI domain: 0 bus: 132 device: 0
GPU id:2 name:Tesla K80 PCI domain: 0 bus: 133 device: 0
selecting device 0
testing size: 65536
rounded size: 65536
device ptr: 2305ba0000
bar_ptr: 0x7f2299166000
info.va: 2305ba0000
info.mapped_size: 65536
info.page_size: 65536
page offset: 0
user-space pointer:0x7f2299166000
BAR writing test, size=65536 offset=0 num_iters=10000
DBG:  sse4_1=1 avx=1 sse=1 sse2=1
DBG:  using AVX implementation of gdr_copy_to_bar
BAR1 write BW: 6812.08MB/s
BAR reading test, size=65536 offset=0 num_iters=100
DBG:  using SSE4_1 implementation of gdr_copy_from_bar
BAR1 read BW: 669.825MB/s
unmapping buffer
unpinning buffer
closing gdrdrv
```


## Bug filing

For reporting issues you may be having using any of NVIDIA software or
reporting suspected bugs we would recommend you use the bug filing system
which is available to NVIDIA registered developers on the developer site.

If you are not a member you can [sign
up](https://developer.nvidia.com/accelerated-computing-developer).

Once a member you can submit issues using [this
form](https://developer.nvidia.com/nvbugs/cuda/add). Be sure to select
GPUDirect in the "Relevant Area" field.

You can later track their progress using the __My Bugs__ link on the left of
this [view](https://developer.nvidia.com/user).

