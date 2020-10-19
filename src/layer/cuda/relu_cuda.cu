#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_util.h"

#include <iostream>


__global__ void gpu_relu_forward_inplace(float* d_input, int input_size, float slope) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; //limited to 1024 rows

    if (index >= input_size) return;

    if (slope == 0.f) {
        d_input[index] = d_input[index] > 0 ? d_input[index] : 0;
    }
    else {
        d_input[index] = d_input[index] > 0 ? d_input[index] : d_input[index] * slope;
    }
}

__global__ void gpu_relu_forward_inplace_int8(int8_t* d_input, int input_size, float slope) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size) return;

    if (slope == 0.f) {
        d_input[index] = d_input[index] > 0 ? d_input[index] : 0;
    }
    else {
        d_input[index] = d_input[index] > 0 ? d_input[index] : d_input[index] * slope;
    }
}

namespace ncnn {

int relu_cuda_forward_inplace(float* d_input, int input_size, float slope)
{
    const int thread_per_block = 512;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);

    gpu_relu_forward_inplace<<<grid_size, block_size>>>(d_input, input_size, slope);

    return 0;
}

int relu_cuda_forward_inplace_int8(int8_t * d_input, int input_size, float slope)
{
    const int thread_per_block = 512;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);

    gpu_relu_forward_inplace_int8<<<grid_size, block_size>>>(d_input, input_size, slope);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return 0;
}

}