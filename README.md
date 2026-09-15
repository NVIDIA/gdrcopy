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
- gdrcopy_sanity, which contains unit tests for the library and the driver.
- gdrcopy_copybw, a minimal application which calculates the R/W bandwidth for a specific buffer size.
- gdrcopy_copylat, a benchmark application which calculates the R/W copy latency for a range of buffer sizes.
- gdrcopy_apiperf, an application for benchmarking the latency of each GDRCopy API call.
- gdrcopy_pplat, a benchmark application which calculates the round-trip ping-pong latency between GPU and CPU.

## Requirements

GPUDirect RDMA requires [NVIDIA Data Center GPU](https://www.nvidia.com/en-us/data-center/) or [NVIDIA RTX GPU](https://www.nvidia.com/en-us/design-visualization/rtx/) (formerly Tesla and Quadro) based on Kepler or newer generations, see [GPUDirect
RDMA](http://developer.nvidia.com/gpudirect).  For more general information,
please refer to the official GPUDirect RDMA [design
document](http://docs.nvidia.com/cuda/gpudirect-rdma).

The device driver requires GPU display driver >= 331.14. The library and tests
require CUDA >= 6.0.

DKMS is a prerequisite for installing GDRCopy kernel module package. On RHEL
or SLE,
however, users have an option to build kmod and install it instead of the DKMS
package. See [Build and installation](#build-and-installation) section for more details.

```shell
# On RHEL
# dkms can be installed from epel-release. See https://fedoraproject.org/wiki/EPEL.
$ sudo yum install dkms

# On Debian - No additional dependency

# On SLE / Leap
# On SLE dkms can be installed from PackageHub.
$ sudo zypper install dkms rpmbuild
```

CUDA and GPU display driver must be installed before building and/or installing GDRCopy.
The installation instructions can be found in https://developer.nvidia.com/cuda-downloads.

GPU display driver header files are also required. They are installed as a part
of the driver (or CUDA) installation with  *runfile*. If you install the driver
via package management, we suggest
- On RHEL, `sudo dnf module install nvidia-driver:latest-dkms`.
- On Debian, `sudo apt install nvidia-dkms-<your-nvidia-driver-version>`.
- On SLE, `sudo zypper install nvidia-gfx<your-nvidia-driver-version>-kmp`.

The supported architectures are Linux x86\_64 and arm64. The supported
platforms are RHEL8, RHEL9, Ubuntu20\_04, Ubuntu22\_04,
SLE-15 (any SP) and Leap 15.x.
Support for the POWER (ppc64le) architecture has been removed. If you need
GDRCopy on POWER, please use version 2.6 or earlier.

Root privileges are necessary to load/install the kernel-mode device
driver.

### DMA-BUF mmap backend

GDRCopy can export GPU memory as a Linux dma-buf via the CUDA driver and map
it into user space with a plain `mmap()` on the dma-buf file descriptor. This
backend does not require the GDRCopy kernel module (`gdrdrv`) and is intended
for environments where `gdrdrv` is not installed or loaded.

**Requirements**

- CUDA driver **13.3 or newer**

**Backend selection**

On `gdr_open()`, GDRCopy tries `gdrdrv` first. If `gdrdrv` is not installed or
fails to open, it falls back to the dma-buf mmap backend (provided the driver supports). To force the dma-buf
backend even when `gdrdrv` is available, set environment variable `GDRCOPY_USE_DMABUF_MMAP=1` before
calling `gdr_open()`. To check at runtime which backend is active:

```c
int using_dmabuf;
gdr_get_attribute(g, GDR_ATTR_USING_DMA_BUF_MMAP, &using_dmabuf);
// using_dmabuf != 0  -> dma-buf mmap backend is in use
```

**Mapping type**

CPU cacheability is decided by the CUDA driver at pin time and cannot be
changed afterwards. The mapping type depends on the pin flag and if the platform is coherent.

| Pin flag                  | Coherent platform           | Non-coherent platform |
|---------------------------|-----------------------------|-----------------------|
| Default                   | `GDR_MAPPING_TYPE_CACHING`  | `GDR_MAPPING_TYPE_WC` |
| `GDR_PIN_FLAG_FORCE_PCIE` | `GDR_MAPPING_TYPE_WC`       | `GDR_MAPPING_TYPE_WC` |

The dmabuf backend does not support user-requested mapping types: the type
the pin produces is the only type `gdr_map_v2` will accept. Passing an
explicit cache flag (`GDR_MAP_FLAG_REQ_CACHE_MAPPING`,
`GDR_MAP_FLAG_REQ_WC_MAPPING`, …) that asks for anything other than the
default returns `EINVAL`.

**Behavior differences vs. the gdrdrv backend**

- **Persistent mappings.** All dma-buf mappings are persistent;
  `GDR_ATTR_USE_PERSISTENT_MAPPING` always returns 1.
- **One fd per pinned buffer.** Each `gdr_pin_buffer` consumes one dma-buf
  file descriptor until `gdr_unpin_buffer`. Applications that pin many
  buffers should account for the process FD limit.
- **No timing fields.** `gdr_get_info_v2` returns `tm_cycles = 0` and
  `cycles_per_ms = 0`.
- **No invalidation callback.** `gdr_get_callback_flag` always returns 0.
- **GDR API compatibility** — Standard GDRCopy APIs remain unchanged


## Build and installation

We provide three ways for building and installing GDRCopy.

### rpm package

```shell
# For RHEL:
$ sudo yum groupinstall 'Development Tools'
$ sudo yum install dkms rpm-build make

# For SLE:
$ sudo zypper in dkms rpmbuild

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
$ sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms
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

### Notes

Compiling the gdrdrv driver requires the NVIDIA driver source code, which is typically installed at
`/usr/src/nvidia-<version>`. Our make file automatically detects and picks that source code. In case there are multiple
versions installed, it is possible to pass the correct path by defining the NVIDIA_SRC_DIR variable, e.g. `export
NVIDIA_SRC_DIR=/usr/src/nvidia-520.61.05/nvidia` before building the gdrdrv module.

There are two major flavors of NVIDIA driver: 1) proprietary, and 2)
[opensource](https://developer.nvidia.com/blog/nvidia-releases-open-source-gpu-kernel-modules/). We detect the flavor
when compiling gdrdrv based on the source code of the NVIDIA driver. Different flavors come with different features and
restrictions:
- gdrdrv compiled with the opensource flavor will provide functionality and high performance on all platforms. However,
  you will not be able to load this gdrdrv driver when the proprietary NVIDIA driver is loaded.
- gdrdrv compiled with the proprietary flavor can always be loaded regardless of the flavor of NVIDIA driver you have
  loaded. However, it may have suboptimal performance on coherent platforms such as Grace-Hopper. Functionally, it will not
  work correctly on Intel CPUs with Linux kernel built with confidential compute (CC) support, i.e.
  `CONFIG_ARCH_HAS_CC_PLATFORM=y`, *WHEN* CC is enabled at runtime.


## Tests

Execute provided tests:
```shell
$ gdrcopy_sanity 
Total: 36, Passed: 31, Failed: 0, Waived: 5

List of waived tests:
    basic_v2_forcepci_cumemalloc
    basic_v2_forcepci_vmmalloc
    basic_with_tokens
    data_validation_mix_mappings_cumemalloc
    data_validation_v2_forcepci_cumemalloc


$ gdrcopy_copybw
GPU id:0; name: NVIDIA B200; Bus id: 0000:1b:00
GPU id:1; name: NVIDIA B200; Bus id: 0000:43:00
GPU id:2; name: NVIDIA B200; Bus id: 0000:52:00
GPU id:3; name: NVIDIA B200; Bus id: 0000:61:00
GPU id:4; name: NVIDIA B200; Bus id: 0000:9d:00
GPU id:5; name: NVIDIA B200; Bus id: 0000:c3:00
GPU id:6; name: NVIDIA B200; Bus id: 0000:d1:00
GPU id:7; name: NVIDIA B200; Bus id: 0000:df:00
selecting device 0
testing size: 131072
rounded size: 131072
gpu alloc fn: cuMemAlloc
device ptr: 7df86b000000
use force pcie: no
map_d_ptr: 0x7dfa9009c000
info.va: 7df86b000000
info.mapped_size: 131072
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer:0x7dfa9009c000
store fences: enabled
writing test, size=131072 offset=0 num_iters=10000
write BW: median 22360.8MB/s, min 20661.2MB/s
reading test, size=131072 offset=0 num_iters=100
read BW: median 890.824MB/s, min 841.881MB/s
unmapping buffer
unpinning buffer
closing gdrdrv


$ gdrcopy_copylat
GPU id:0; name: NVIDIA B200; Bus id: 0000:1b:00
GPU id:1; name: NVIDIA B200; Bus id: 0000:43:00
GPU id:2; name: NVIDIA B200; Bus id: 0000:52:00
GPU id:3; name: NVIDIA B200; Bus id: 0000:61:00
GPU id:4; name: NVIDIA B200; Bus id: 0000:9d:00
GPU id:5; name: NVIDIA B200; Bus id: 0000:c3:00
GPU id:6; name: NVIDIA B200; Bus id: 0000:d1:00
GPU id:7; name: NVIDIA B200; Bus id: 0000:df:00
selecting device 0
device ptr: 0x73aa1c800000
allocated size: 16777216
gpu alloc fn: cuMemAlloc
use force pcie: no

map_d_ptr: 0x73a9f3000000
info.va: 73aa1c800000
info.mapped_size: 16777216
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer: 0x73a9f3000000
use cold cache: no
store fences: enabled
load fences (gdr_copy_from_mapping): enabled

gdr_copy_to_mapping num iters for each size: 10000
WARNING: Measuring the API invocation overhead as observed by the CPU. Data might not be ordered all the way to the GPU internal visibility.
Test 			 Size(B) 	 Median Time(us) 	 Min. Time(us)
gdr_copy_to_mapping 	        1 	      0.1014 	      0.1011
gdr_copy_to_mapping 	        2 	      0.1015 	      0.0938
gdr_copy_to_mapping 	        4 	      0.1014 	      0.0938
gdr_copy_to_mapping 	        8 	      0.1015 	      0.0938
gdr_copy_to_mapping 	       16 	      0.1015 	      0.0969
gdr_copy_to_mapping 	       32 	      0.1016 	      0.0976
gdr_copy_to_mapping 	       64 	      0.1030 	      0.1010
gdr_copy_to_mapping 	      128 	      0.1043 	      0.0977
gdr_copy_to_mapping 	      256 	      0.1100 	      0.1038
gdr_copy_to_mapping 	      512 	      0.1312 	      0.1256
gdr_copy_to_mapping 	     1024 	      0.1800 	      0.1741
gdr_copy_to_mapping 	     2048 	      0.2042 	      0.1994
gdr_copy_to_mapping 	     4096 	      0.2730 	      0.2718
gdr_copy_to_mapping 	     8192 	      0.4457 	      0.4450
gdr_copy_to_mapping 	    16384 	      0.7575 	      0.7554
gdr_copy_to_mapping 	    32768 	      1.3806 	      1.3779
gdr_copy_to_mapping 	    65536 	      2.8588 	      2.8558
gdr_copy_to_mapping 	   131072 	      5.5994 	      5.5858
gdr_copy_to_mapping 	   262144 	     11.0404 	     11.0303
gdr_copy_to_mapping 	   524288 	     21.9059 	     21.8922
gdr_copy_to_mapping 	  1048576 	     43.6844 	     43.6310
gdr_copy_to_mapping 	  2097152 	     91.8368 	     91.7010
gdr_copy_to_mapping 	  4194304 	    209.8323 	    209.6777
gdr_copy_to_mapping 	  8388608 	    419.9362 	    419.4436
gdr_copy_to_mapping 	 16777216 	    839.3445 	    838.9237

gdr_copy_from_mapping num iters for each size: 100
Test 			 Size(B) 	 Median Time(us) 	 Min. Time(us)
gdr_copy_from_mapping 	        1 	      1.1600 	      1.1540
gdr_copy_from_mapping 	        2 	      1.1600 	      1.1300
gdr_copy_from_mapping 	        4 	      1.1590 	      1.1480
gdr_copy_from_mapping 	        8 	      1.1590 	      1.1270
gdr_copy_from_mapping 	       16 	      1.1590 	      1.1240
gdr_copy_from_mapping 	       32 	      1.1590 	      1.1250
gdr_copy_from_mapping 	       64 	      1.1590 	      1.1370
gdr_copy_from_mapping 	      128 	      1.1670 	      1.1470
gdr_copy_from_mapping 	      256 	      1.1700 	      1.1490
gdr_copy_from_mapping 	      512 	      1.1890 	      1.1640
gdr_copy_from_mapping 	     1024 	      2.3940 	      2.3620
gdr_copy_from_mapping 	     2048 	      2.3700 	      2.3510
gdr_copy_from_mapping 	     4096 	      4.6920 	      4.6400
gdr_copy_from_mapping 	     8192 	      8.9220 	      8.6870
gdr_copy_from_mapping 	    16384 	     17.6940 	     17.3350
gdr_copy_from_mapping 	    32768 	     35.7300 	     34.7390
gdr_copy_from_mapping 	    65536 	     70.5915 	     69.6220
gdr_copy_from_mapping 	   131072 	    140.2520 	    138.9910
gdr_copy_from_mapping 	   262144 	    279.8475 	    278.2950
gdr_copy_from_mapping 	   524288 	    562.5135 	    555.5700
gdr_copy_from_mapping 	  1048576 	   1121.5775 	   1116.1780
gdr_copy_from_mapping 	  2097152 	   2242.2870 	   2234.7010
gdr_copy_from_mapping 	  4194304 	   4485.6935 	   4473.8390
gdr_copy_from_mapping 	  8388608 	   8970.7600 	   8956.1840
gdr_copy_from_mapping 	 16777216 	  17942.0570 	  17914.3680
unmapping buffer
unpinning buffer
closing gdrdrv


$ gdrcopy_apiperf -s 8
GPU id:0; name: NVIDIA B200; Bus id: 0000:1b:00
GPU id:1; name: NVIDIA B200; Bus id: 0000:43:00
GPU id:2; name: NVIDIA B200; Bus id: 0000:52:00
GPU id:3; name: NVIDIA B200; Bus id: 0000:61:00
GPU id:4; name: NVIDIA B200; Bus id: 0000:9d:00
GPU id:5; name: NVIDIA B200; Bus id: 0000:c3:00
GPU id:6; name: NVIDIA B200; Bus id: 0000:d1:00
GPU id:7; name: NVIDIA B200; Bus id: 0000:df:00
selecting device 0
device ptr: 0x7a8a6b000000
allocated size: 65536
Stat	Size(B)	pin.Time(us)	map.Time(us)	get_info.Time(us)	unmap.Time(us)	unpin.Time(us)
median	65536	82.119500	3.713000	0.175000	3.922500	29.251000
min	65536	79.159000	3.570000	0.172000	3.855000	28.218000
Histogram of gdr_pin_buffer latency for 65536 bytes
[79.159000	-	158.318000]	8
[158.318000	-	237.477000]	9
[237.477000	-	316.636000]	34
[316.636000	-	395.795000]	40
[395.795000	-	474.954000]	5
[474.954000	-	554.113000]	3
[554.113000	-	633.272000]	0
[633.272000	-	712.431000]	0
[712.431000	-	791.590000]	0
[791.590000	-	870.749000]	0

closing gdrdrv



$ numactl -N 1 -l gdrcopy_pplat
GPU id:0; name: NVIDIA B200; Bus id: 0000:1b:00
GPU id:1; name: NVIDIA B200; Bus id: 0000:43:00
GPU id:2; name: NVIDIA B200; Bus id: 0000:52:00
GPU id:3; name: NVIDIA B200; Bus id: 0000:61:00
GPU id:4; name: NVIDIA B200; Bus id: 0000:9d:00
GPU id:5; name: NVIDIA B200; Bus id: 0000:c3:00
GPU id:6; name: NVIDIA B200; Bus id: 0000:d1:00
GPU id:7; name: NVIDIA B200; Bus id: 0000:df:00
selecting device 0
use force pcie: no
We will measure the visibility of the flag value only. Setting nblocks and nthreads to 1.
Benchmark mode: CPU produces and GPU consumes
gpu alloc fn: cuMemAlloc
Measuring the visibility latency of the flag value.
Running 1000 iterations with flag size 4 bytes.

CPU writes to gpu_flag. GPU polls on the expected gpu_flag value. GPU writes back via cpu_flag. CPU polls on the expected cpu_flag value. We report the round-trip time from when CPU writes to gpu_flag until it observes the update in cpu_flag.
CPU does the time measurement.

Round-trip latency per iteration is (min) 2.2055, (median) 2.2139 us
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
$ numactl -N 0 -l gdrcopy_copybw -d 0 -s $((64 * 1024)) -o $((0 * 1024)) -c $((64 * 1024))
GPU id:0; name: NVIDIA B200; Bus id: 0000:1b:00
GPU id:1; name: NVIDIA B200; Bus id: 0000:43:00
GPU id:2; name: NVIDIA B200; Bus id: 0000:52:00
GPU id:3; name: NVIDIA B200; Bus id: 0000:61:00
GPU id:4; name: NVIDIA B200; Bus id: 0000:9d:00
GPU id:5; name: NVIDIA B200; Bus id: 0000:c3:00
GPU id:6; name: NVIDIA B200; Bus id: 0000:d1:00
GPU id:7; name: NVIDIA B200; Bus id: 0000:df:00
selecting device 0
testing size: 65536
rounded size: 65536
gpu alloc fn: cuMemAlloc
device ptr: 73de17000000
use force pcie: no
map_d_ptr: 0x73de29163000
info.va: 73de17000000
info.mapped_size: 65536
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer:0x73de29163000
store fences: enabled
writing test, size=65536 offset=0 num_iters=10000
write BW: median 22863.4MB/s, min 22336.3MB/s
reading test, size=65536 offset=0 num_iters=100
read BW: median 884.505MB/s, min 802.218MB/s
unmapping buffer
unpinning buffer
closing gdrdrv
```

or on the other socket:
```shell
$ numactl -N 1 -l gdrcopy_copybw -d 0 -s $((64 * 1024)) -o $((0 * 1024)) -c $((64 * 1024))
GPU id:0; name: NVIDIA B200; Bus id: 0000:1b:00
GPU id:1; name: NVIDIA B200; Bus id: 0000:43:00
GPU id:2; name: NVIDIA B200; Bus id: 0000:52:00
GPU id:3; name: NVIDIA B200; Bus id: 0000:61:00
GPU id:4; name: NVIDIA B200; Bus id: 0000:9d:00
GPU id:5; name: NVIDIA B200; Bus id: 0000:c3:00
GPU id:6; name: NVIDIA B200; Bus id: 0000:d1:00
GPU id:7; name: NVIDIA B200; Bus id: 0000:df:00
selecting device 0
testing size: 65536
rounded size: 65536
gpu alloc fn: cuMemAlloc
device ptr: 7d660b000000
use force pcie: no
map_d_ptr: 0x7d661cf37000
info.va: 7d660b000000
info.mapped_size: 65536
info.page_size: 65536
info.mapped: 1
info.wc_mapping: 1
page offset: 0
user-space pointer:0x7d661cf37000
store fences: enabled
writing test, size=65536 offset=0 num_iters=10000
write BW: median 22047.8MB/s, min 21492.4MB/s
reading test, size=65536 offset=0 num_iters=100
read BW: median 829.468MB/s, min 724.512MB/s
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

If gdrdrv is compiled with the proprietary flavor of NVIDIA driver, GDRCopy does not fully support Linux with the
confidential computing (CC) configuration with Intel CPU. In particular, it does not functional if
`CONFIG_ARCH_HAS_CC_PLATFORM=y` and CC is enabled at runtime. However, it works if CC is disabled or
`CONFIG_ARCH_HAS_CC_PLATFORM=n`. This issue is not applied to AMD CPU. To avoid this issue, please compile and load
gdrdrv with the opensource flavor of NVIDIA driver.

On open-source NVIDIA driver builds, and on proprietary builds with Linux
before 6.15, `gdr_map()` sets `VM_DONTCOPY` so mappings are not inherited across
`fork()`.

On proprietary NVIDIA driver builds with Linux 6.15 or later, `VM_DONTCOPY`
cannot be set because vm_flags_set is GPL protected. Mappings from `gdr_map()` 
are therefore inherited by children across `fork()`. The child's copy is not 
tracked by the parent's `gdr_unmap()` / `gdr_unpin_buffer()`, and tearing it down 
can break the parent's invalidation tracking. Detect this at run time with
`GDR_ATTR_VMA_INHERITED_ON_FORK` via `gdr_get_attribute()`, or with
`cat /proc/driver/gdrdrv/params`. The `invalidation_fork_after_gdr_map_*`
tests in `gdrcopy_sanity` are waived in that configuration.

To allow the loading of unsupported 3rd party modules in SLE, set `allow_unsupported_modules 1` in
/etc/modprobe.d/unsupported-modules. After making this change, modules missing the "supported" flag, will be allowed to
load.


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
