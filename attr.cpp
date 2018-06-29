#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include "cuda.h"

using namespace std;
#include "common.hpp"

int main(int argc, char *argv[])
{
    void *dummy;
    CUdevice dev;
    ASSERTDRV(cuInit(0));
    ASSERTDRV(cuDeviceGet(&dev, 0));

#define DEF(S) { #S, S }
 
    struct { const char* s; CUdevice_attribute e; } attrs[] = {
      DEF(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
      DEF(CU_DEVICE_ATTRIBUTE_ECC_ENABLED),
      DEF(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED),
      DEF(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS),
      DEF(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS),
      DEF(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES),
      DEF(CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST),
      DEF(CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES),
      DEF(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH),
      DEF(CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH),
      DEF(CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS),
      DEF(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR),
      DEF(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS),
      DEF(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM),
      DEF(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED)
    };

    for (int n=0; n<sizeof(attrs)/sizeof(attrs[0]); ++n) {
      int flag;
      ASSERTDRV(cuDeviceGetAttribute(&flag, attrs[n].e, dev));
      printf("attr:%s value=%d\n", attrs[n].s, flag);
    }

}
