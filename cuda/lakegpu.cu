#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define __DEBUG

#define VSQR 0.1
#define TSCALE 1.0

#define CUDA_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__, __LINE__)

/**************************************
 * void __cudaSafeCall(cudaError err, const char *file, const int line)
 * void __cudaCheckError(const char *file, const int line)
 *
 * These routines were taken from the GPU Computing SDK
 * (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
 **************************************/
inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
#ifdef __DEBUG

#pragma warning(push)
#pragma warning(disable : 4127) // Prevent warning on do-while(0);
    do {
        if (cudaSuccess != err) {
            fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#pragma warning(pop)
#endif // __DEBUG
    return;
}

inline void __cudaCheckError(const char* file, const int line) {
#ifdef __DEBUG
#pragma warning(push)
#pragma warning(disable : 4127) // Prevent warning on do-while(0);
    do {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
            exit(-1);
        }
        // More careful checking. However, this will affect performance.
        // Comment if not needed.
        /*err = cudaThreadSynchronize();
        if( cudaSuccess != err )
        {
          fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
                   file, line, cudaGetErrorString( err ) );
          exit( -1 );
        }*/
    } while (0);
#pragma warning(pop)
#endif // __DEBUG
    return;
}

int tpdt(double* t, double dt, double end_time);
__device__ double f_gpu(double p, double t) { return -expf(-TSCALE * t) * p; }

__global__ void evolve_gpu(double* un, double* uc, double* uo, double* pebbles, int n, double h, double dt, double t)
{
    __shared__ int i, j, idx;

    idx = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);

    i = idx / n;
    j = idx % n;

    if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
        un[idx] = 0.;
    else
        un[idx] = 2 * uc[idx] - uo[idx] + VSQR * (dt * dt) * ((uc[idx - 1] + uc[idx + 1] + uc[idx + n] + uc[idx - n] + 0.25 * (uc[idx - 1 - n] + uc[idx - 1 + n] + uc[idx + 1 - n] + uc[idx + 1 + n]) - 5 * uc[idx]) / (h * h) + f_gpu(pebbles[idx], t));
}

void run_gpu(double* u, double* u0, double* u1, double* pebbles, int n, double h, double end_time, int nthreads) {
    cudaEvent_t kstart, kstop;
    float ktime;

    /* HW2: Define your local variables here */
    // declare the variables
    double* gpu_uc;
    double* gpu_uo;
    double* gpu_un;
    double* gpu_pebbles;

    /* Set up device timers */
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaEventCreate(&kstart));
    CUDA_CALL(cudaEventCreate(&kstop));

    /* HW2: Add CUDA kernel call preperation code here */
    // malloc mem on GPU and copy the content
    cudaMalloc((void**)&gpu_uc, sizeof(double) * n * n);
    cudaMalloc((void**)&gpu_uo, sizeof(double) * n * n);
    cudaMalloc((void**)&gpu_un, sizeof(double) * n * n);
    cudaMalloc((void**)&gpu_pebbles, sizeof(double) * n * n);
    cudaMemcpy((void*)gpu_uo, (void*)u0, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_uc, (void*)u1, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)gpu_pebbles, (void*)pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    double t = 0., dt = h / 2.;

    int grid_size = n / nthreads;
    int block_size = nthreads;

    dim3 grid(grid_size, grid_size);
    dim3 block(block_size, block_size);

    /* Start GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstart, 0));

    /* HW2: Add main lake simulation loop here */
    // do the calculation
    while (1) {
        evolve_gpu<<<grid, block>>>(gpu_un, gpu_uc, gpu_uo, gpu_pebbles, n, h, dt, t);

        cudaMemcpy((void*)gpu_uo, (void*)gpu_uc, sizeof(double) * n * n, cudaMemcpyDeviceToDevice);
        cudaMemcpy((void*)gpu_uc, (void*)gpu_un, sizeof(double) * n * n, cudaMemcpyDeviceToDevice);

        if (!tpdt(&t, dt, end_time))
            break;
    }
    
    cudaMemcpy((void*)u, (void*)gpu_un, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    /* Stop GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstop, 0));
    CUDA_CALL(cudaEventSynchronize(kstop));
    CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
    printf("GPU computation: %f msec\n", ktime);

    /* HW2: Add post CUDA kernel call processing and cleanup here */

    /* timer cleanup */
    CUDA_CALL(cudaEventDestroy(kstart));
    CUDA_CALL(cudaEventDestroy(kstop));
}
