//
// Created by amarko on 9/18/20.
//

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

#endif //NCNN_CUDA_UTIL_H

#endif
