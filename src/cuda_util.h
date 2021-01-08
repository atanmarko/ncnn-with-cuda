
#if NCNN_CUDA

#ifndef NCNN_CUDA_UTIL_H
#define NCNN_CUDA_UTIL_H


#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#if defined(__cplusplus) && defined(__CUDACC__)

static __device__ void cuda_lock(int* _mutex)
{
    while (atomicCAS(_mutex, 0, 1) != 0)
        ;
}

static __device__ void cuda_unlock(int* _mutex)
{
    atomicExch(_mutex, 0);
}

#endif

#endif //NCNN_CUDA_UTIL_H

#endif
