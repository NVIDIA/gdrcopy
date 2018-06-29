#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define __CUDACHECK(stmt, cond_str)					\
	do {								\
		cudaError_t result = (stmt);				\
		if (cudaSuccess != result) {				\
			fprintf(stderr, "Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", cond_str, __FILE__, __LINE__, result, cudaGetErrorString(result)); \
			exit(EXIT_FAILURE);				\
		}							\
        } while (0)

#define CUDACHECK(stmt) __CUDACHECK(stmt, #stmt)

extern int gpu_clock_rate;
typedef unsigned long long ns_t;

static __device__ ns_t getTimerNs()
{
        unsigned long long time = 0;
        asm("mov.u64  %0, %globaltimer;" : "=l"(time) );
        //time = clock();
        return (ns_t)time;
}

__global__ void polling_kernel(ns_t wait_ns)
{
        if (threadIdx.x == 0) {
                ns_t start = getTimerNs();
                ns_t now;
                do {
                        now = getTimerNs();
                } while ((long long)now - (long long)start < wait_ns);
        }
        __syncthreads();
}

int run_polling_kernel(CUstream stream, unsigned long long wait_ns)
{
        const int nblocks = 1;
        const int nthreads = 32;
        polling_kernel <<<nblocks, nthreads, 0, stream>>> (wait_ns);
        CUDACHECK(cudaGetLastError());
        return 0;
}
