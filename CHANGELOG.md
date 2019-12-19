# Changelog

## [2.0] - 2019-09-16
- Harden security in gdrdrv.
- Enable cached mappings in POWER9.
- Improve copy performance with unrolling in POWERPC.
- Creates _sanity_ unit test for testing the functionality and security.
- Consolidate _basic_ and _validate_ into _sanity_ unit test.
- Introduce compile time and runtime version checking in _libgdrapi_.
- Improve rpm packaging.
- Introduce deb packaging for the userspace library and the applications.
- Introduce dkms packaging for the _gdrdrv_ driver.
- Rename gdr\_copy\_from/to\_bar to gdr\_copy\_from/to\_mapping.
- Update README

## [1.3] - 2018-07-26
- Add _gdrdrv_ driver for converting cudaMalloc'd addresses to the GPU's BAR1
  addresses and exposing them to CPU-accessible virtual addresses.
- Add _libgdrapi_, a user-space library for communicating with the gdrdrv driver.
- Add _basic_ application as an minimal example on how to use gdrcopy.
- Add _copybw_ application as a complete example on how CPU could read/write to
  cudaMalloc'd memory via BAR1 mappings.
- Add _validate_ unit test to ensure that gdrcopy functions as expected.
- Add a script for packaging gdrcopy in the rpm format.


[2.0]: https://github.com/NVIDIA/gdrcopy/releases/tag/v2.0
[1.3]: https://github.com/NVIDIA/gdrcopy/releases/tag/v1.3

