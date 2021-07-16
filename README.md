# GDRCopy

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

The library comes with a few tests like:
- sanity, which contains unit tests for the library and the driver.
- copybw, a minimal application which calculates the R/W bandwidth for a specific buffer size.
- copylat, a benchmark application which calculates the R/W copy latency for a range of buffer sizes.

## Requirements

GPUDirect RDMA requires NVIDIA Tesla or Quadro class GPUs based on Kepler,
Pascal, Volta, or Turing, see [GPUDirect
RDMA](http://developer.nvidia.com/gpudirect).  For more technical informations,
please refer to the official GPUDirect RDMA [design
document](http://docs.nvidia.com/cuda/gpudirect-rdma).

The device driver requires GPU display driver >= 418.40 on ppc64le and >= 331.14 on other platforms. The library and tests
require CUDA >= 6.0. Additionally, the _sanity_ test requires check >= 0.9.8 and
subunit.

DKMS is a prerequisite for installing GDRCopy kernel module package. On RHEL,
however, users have an option to build kmod and install it instead of the DKMS
package. See [Build and installation](#build-and-installation) section for more details.

```shell
# On RHEL
# dkms can be installed from epel-release. See https://fedoraproject.org/wiki/EPEL.
$ sudo yum install dkms check check-devel subunit subunit-devel

# On Debian
$ sudo apt install check libsubunit0 libsubunit-dev
```

CUDA and GPU display driver must be installed before building and/or installing GDRCopy.
The installation instructions can be found in https://developer.nvidia.com/cuda-downloads.

GPU display driver header files are also required. They are installed as a part
of the driver (or CUDA) installation with  *runfile*. If you install the driver
via package management, we suggest
- On RHEL, `sudo dnf module install nvidia-driver:latest-dkms`.
- On Debian, `sudo apt install nvidia-dkms-<your-nvidia-driver-version>`.

The supported architectures are Linux x86_64, ppc64le, and arm64. The supported
platforms are RHEL7, RHEL8, Ubuntu16_04, Ubuntu18_04, and Ubuntu20_04.

Root privileges are necessary to load/install the kernel-mode device
driver.


## Build and installation

We provide three ways for building and installing GDRCopy.

### rpm package

```shell
$ sudo yum groupinstall 'Development Tools'
$ sudo yum install dkms rpm-build make check check-devel subunit subunit-devel
$ cd packages
$ CUDA=<cuda-install-top-dir> ./build-rpm-packages.sh
$ sudo rpm -Uvh gdrcopy-kmod-<version>dkms.noarch.<platform>.rpm
$ sudo rpm -Uvh gdrcopy-<version>.<arch>.<platform>.rpm
$ sudo rpm -Uvh gdrcopy-devel-<version>.noarch.<platform>.rpm
```
DKMS package is the default kernel module package that `build-rpm-packages.sh`
generates. To create kmod package, `-m` option must be passed to the script.
Unlike the DKMS package, the kmod package contains a prebuilt GDRCopy kernel
module which is specific to the NVIDIA driver version and the Linux kernel
version used to build it.


### deb package

```shell
$ sudo apt install build-essential devscripts debhelper check libsubunit-dev fakeroot pkg-config dkms
$ cd packages
$ CUDA=<cuda-install-top-dir> ./build-deb-packages.sh
$ sudo dpkg -i gdrdrv-dkms_<version>_<arch>.<platform>.deb
$ sudo dpkg -i libgdrapi_<version>_<arch>.<platform>.deb
$ sudo dpkg -i gdrcopy-tests_<version>_<arch>.<platform>.deb
$ sudo dpkg -i gdrcopy_<version>_<arch>.<platform>.deb
```

### from source

```shell
$ make prefix=<install-to-this-location> CUDA=<cuda-install-top-dir> all install
$ sudo ./insmod.sh
```

If `libcheck` is installed in a non-standard path and therefore is not
picked by `pkg-config`, you can set the `PKG_CONFIG_PATH` environment
variable to the directory which contains the `check.pc` file and pass it
down to make:
```shell
$ PKG_CONFIG_PATH=/check_install_path/lib/pkgconfig/ make <...>
```

## Tests

Execute provided tests:
```shell
$ sanity 
Running suite(s): Sanity
100%: Checks: 27, Failures: 0, Errors: 0


$ copybw
GPU id:0; name: Tesla V100-SXM2-32GB; Bus id: 0000:06:00
GPU id:1; name: Tesla V100-SXM2-32GB; Bus id: 0000:07:00
GPU id:2; name: Tesla V100-SXM2-32GB; Bus id: 0000:0a:00
GPU id:3; name: Tesla V100-SXM2-32GB; Bus id: 0000:0b:00
GPU id:4; name: Tesla V100-SXM2-32GB; Bus id: 0000:85:00
GPU id:5; name: Tesla V100-SXM2-32GB; Bus id: 0000:86:00
GPU id:6; name: Tesla V100-SXM2-32GB; Bus id: 0000:89:00
GPU id:7; name: Tesla V100-SXM2-32GB; Bus id: 0000:8a:00
selecting device 0
testing size: 131072
rounded size: 131072
gpu alloc fn: cuMemAlloc
device ptr: 7f1153a00000
map_d_ptr: 0x7f1172257000
info.va: 7f1153a00000
info.mapped_size: 131072
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer:0x7f1172257000
writing test, size=131072 offset=0 num_iters=10000
write BW: 9638.54MB/s
reading test, size=131072 offset=0 num_iters=100
read BW: 530.135MB/s
unmapping buffer
unpinning buffer
closing gdrdrv


$ copylat
GPU id:0; name: Tesla V100-SXM2-32GB; Bus id: 0000:06:00
GPU id:1; name: Tesla V100-SXM2-32GB; Bus id: 0000:07:00
GPU id:2; name: Tesla V100-SXM2-32GB; Bus id: 0000:0a:00
GPU id:3; name: Tesla V100-SXM2-32GB; Bus id: 0000:0b:00
GPU id:4; name: Tesla V100-SXM2-32GB; Bus id: 0000:85:00
GPU id:5; name: Tesla V100-SXM2-32GB; Bus id: 0000:86:00
GPU id:6; name: Tesla V100-SXM2-32GB; Bus id: 0000:89:00
GPU id:7; name: Tesla V100-SXM2-32GB; Bus id: 0000:8a:00
selecting device 0
device ptr: 0x7fa2c6000000
allocated size: 16777216
gpu alloc fn: cuMemAlloc

map_d_ptr: 0x7fa2f9af9000
info.va: 7fa2c6000000
info.mapped_size: 16777216
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer: 0x7fa2f9af9000

gdr_copy_to_mapping num iters for each size: 10000
WARNING: Measuring the API invocation overhead as observed by the CPU. Data
might not be ordered all the way to the GPU internal visibility.
Test             Size(B)     Avg.Time(us)
gdr_copy_to_mapping             1         0.0889
gdr_copy_to_mapping             2         0.0884
gdr_copy_to_mapping             4         0.0884
gdr_copy_to_mapping             8         0.0884
gdr_copy_to_mapping            16         0.0905
gdr_copy_to_mapping            32         0.0902
gdr_copy_to_mapping            64         0.0902
gdr_copy_to_mapping           128         0.0952
gdr_copy_to_mapping           256         0.0983
gdr_copy_to_mapping           512         0.1176
gdr_copy_to_mapping          1024         0.1825
gdr_copy_to_mapping          2048         0.2549
gdr_copy_to_mapping          4096         0.4366
gdr_copy_to_mapping          8192         0.8141
gdr_copy_to_mapping         16384         1.6155
gdr_copy_to_mapping         32768         3.2284
gdr_copy_to_mapping         65536         6.4906
gdr_copy_to_mapping        131072        12.9761
gdr_copy_to_mapping        262144        25.9459
gdr_copy_to_mapping        524288        51.9100
gdr_copy_to_mapping       1048576       103.8028
gdr_copy_to_mapping       2097152       207.5990
gdr_copy_to_mapping       4194304       415.2856
gdr_copy_to_mapping       8388608       830.6355
gdr_copy_to_mapping      16777216      1661.3285

gdr_copy_from_mapping num iters for each size: 100
Test             Size(B)     Avg.Time(us)
gdr_copy_from_mapping           1         0.9069
gdr_copy_from_mapping           2         1.7170
gdr_copy_from_mapping           4         1.7169
gdr_copy_from_mapping           8         1.7164
gdr_copy_from_mapping          16         0.8601
gdr_copy_from_mapping          32         1.7024
gdr_copy_from_mapping          64         3.1016
gdr_copy_from_mapping         128         3.4944
gdr_copy_from_mapping         256         3.6400
gdr_copy_from_mapping         512         2.4394
gdr_copy_from_mapping        1024         2.8022
gdr_copy_from_mapping        2048         4.6615
gdr_copy_from_mapping        4096         7.9783
gdr_copy_from_mapping        8192        14.9209
gdr_copy_from_mapping       16384        28.9571
gdr_copy_from_mapping       32768        56.9373
gdr_copy_from_mapping       65536       114.1008
gdr_copy_from_mapping      131072       234.9382
gdr_copy_from_mapping      262144       496.4011
gdr_copy_from_mapping      524288       985.5196
gdr_copy_from_mapping     1048576      1970.7057
gdr_copy_from_mapping     2097152      3942.5611
gdr_copy_from_mapping     4194304      7888.9468
gdr_copy_from_mapping     8388608     18361.5673
gdr_copy_from_mapping    16777216     36758.8342
unmapping buffer
unpinning buffer
closing gdrdrv


$ apiperf -s 8
GPU id:0; name: Tesla V100-SXM2-32GB; Bus id: 0000:06:00
GPU id:1; name: Tesla V100-SXM2-32GB; Bus id: 0000:07:00
GPU id:2; name: Tesla V100-SXM2-32GB; Bus id: 0000:0a:00
GPU id:3; name: Tesla V100-SXM2-32GB; Bus id: 0000:0b:00
GPU id:4; name: Tesla V100-SXM2-32GB; Bus id: 0000:85:00
GPU id:5; name: Tesla V100-SXM2-32GB; Bus id: 0000:86:00
GPU id:6; name: Tesla V100-SXM2-32GB; Bus id: 0000:89:00
GPU id:7; name: Tesla V100-SXM2-32GB; Bus id: 0000:8a:00
selecting device 0
device ptr: 0x7f1563a00000
allocated size: 65536
Size(B) pin.Time(us)    map.Time(us)    get_info.Time(us)   unmap.Time(us)
unpin.Time(us)
65536   1346.034060 3.603800    0.340270    4.700930    676.612800
Histogram of gdr_pin_buffer latency for 65536 bytes
[1303.852000    -   2607.704000]    93
[2607.704000    -   3911.556000]    0
[3911.556000    -   5215.408000]    0
[5215.408000    -   6519.260000]    0
[6519.260000    -   7823.112000]    0
[7823.112000    -   9126.964000]    0
[9126.964000    -   10430.816000]   0
[10430.816000   -   11734.668000]   0
[11734.668000   -   13038.520000]   0
[13038.520000   -   14342.372000]   2

closing gdrdrv
```

## NUMA effects

Depending on the platform architecture, like where the GPU are placed in
the PCIe topology, performance may suffer if the processor which is driving
the copy is not the one which is hosting the GPU, for example in a
multi-socket server.

In the example below, GPU ID 0 is hosted by
CPU socket 0. By explicitly playing with the OS process and memory
affinity, it is possible to run the test onto the optimal processor:

```shell
$ numactl -N 0 -l copybw -d 0 -s $((64 * 1024)) -o $((0 * 1024)) -c $((64 * 1024))
GPU id:0; name: Tesla V100-SXM2-32GB; Bus id: 0000:06:00
GPU id:1; name: Tesla V100-SXM2-32GB; Bus id: 0000:07:00
GPU id:2; name: Tesla V100-SXM2-32GB; Bus id: 0000:0a:00
GPU id:3; name: Tesla V100-SXM2-32GB; Bus id: 0000:0b:00
GPU id:4; name: Tesla V100-SXM2-32GB; Bus id: 0000:85:00
GPU id:5; name: Tesla V100-SXM2-32GB; Bus id: 0000:86:00
GPU id:6; name: Tesla V100-SXM2-32GB; Bus id: 0000:89:00
GPU id:7; name: Tesla V100-SXM2-32GB; Bus id: 0000:8a:00
selecting device 0
testing size: 65536
rounded size: 65536
gpu alloc fn: cuMemAlloc
device ptr: 7f5817a00000
map_d_ptr: 0x7f583b186000
info.va: 7f5817a00000
info.mapped_size: 65536
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer:0x7f583b186000
writing test, size=65536 offset=0 num_iters=1000
write BW: 9768.3MB/s
reading test, size=65536 offset=0 num_iters=1000
read BW: 548.423MB/s
unmapping buffer
unpinning buffer
closing gdrdrv
```

or on the other socket:
```shell
$ numactl -N 1 -l copybw -d 0 -s $((64 * 1024)) -o $((0 * 1024)) -c $((64 * 1024))
GPU id:0; name: Tesla V100-SXM2-32GB; Bus id: 0000:06:00
GPU id:1; name: Tesla V100-SXM2-32GB; Bus id: 0000:07:00
GPU id:2; name: Tesla V100-SXM2-32GB; Bus id: 0000:0a:00
GPU id:3; name: Tesla V100-SXM2-32GB; Bus id: 0000:0b:00
GPU id:4; name: Tesla V100-SXM2-32GB; Bus id: 0000:85:00
GPU id:5; name: Tesla V100-SXM2-32GB; Bus id: 0000:86:00
GPU id:6; name: Tesla V100-SXM2-32GB; Bus id: 0000:89:00
GPU id:7; name: Tesla V100-SXM2-32GB; Bus id: 0000:8a:00
selecting device 0
testing size: 65536
rounded size: 65536
gpu alloc fn: cuMemAlloc
device ptr: 7fbb63a00000
map_d_ptr: 0x7fbb82ab0000
info.va: 7fbb63a00000
info.mapped_size: 65536
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer:0x7fbb82ab0000
writing test, size=65536 offset=0 num_iters=1000
write BW: 9224.36MB/s
reading test, size=65536 offset=0 num_iters=1000
read BW: 521.262MB/s
unmapping buffer
unpinning buffer
closing gdrdrv
```


## Restrictions and known issues

GDRCopy works with regular CUDA device memory only, as returned by cudaMalloc.
In particular, it does not work with CUDA managed memory.

`gdr_pin_buffer()` accepts any addresses returned by cudaMalloc and its family.
In contrast, `gdr_map()` requires that the pinned address is aligned to the GPU page.
Neither CUDA Runtime nor Driver APIs guarantees that GPU memory allocation
functions return aligned addresses. Users are responsible for proper alignment
of addresses passed to the library.

Two cudaMalloc'd memory regions may be contiguous. Users may call
`gdr_pin_buffer` and `gdr_map` with address and size that extend across these
two regions. This use case is not well-supported in GDRCopy. On rare occassions,
users may experience 1.) an error in `gdr_map`, or 2.) low copy performance
because `gdr_map` cannot provide write-combined mapping.

In some GPU driver versions, pinning the same GPU address multiple times
consumes additional BAR1 space. This is because the space is not properly
reused. If you encounter this issue, we suggest that you try the latest version
of NVIDIA GPU driver.

On POWER9 where CPU and GPU are connected via NVLink, CUDA9.2 and GPU Driver
v396.37 are the minimum requirements in order to achieve the full performance.
GDRCopy works with ealier CUDA and GPU driver versions but the achievable
bandwidth is substantially lower.


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

## Acknowledgment

If you find this software useful in your work, please cite:
R. Shi et al., "Designing efficient small message transfer mechanism for inter-node MPI communication on InfiniBand GPU clusters," 2014 21st International Conference on High Performance Computing (HiPC), Dona Paula, 2014, pp. 1-10, doi: 10.1109/HiPC.2014.7116873.
