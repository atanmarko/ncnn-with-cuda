#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_util.h"
#include "mat.h"

#include <iostream>


__global__ void gpu_bias_forward_inplace(float* a_input, const ncnn::CudaMatInfo a_info, const float* bias) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x; //limited to 1024 rows
    const int input_size = a_info.c * a_info.cstep;
    if (index >= input_size) return;

    const int channel = index / a_info.cstep;
    a_input[index] = a_input[index] + bias[channel];
}

namespace ncnn {

int bias_cuda_forward_inplace(float* a_input, const ncnn::CudaMatInfo& a_info, const float* bias)
{
    int thread_per_block = ((a_info.total_size() / 32) + 1) * 32;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size(a_info.total_size() / thread_per_block + 1, 1, 1);

    gpu_bias_forward_inplace<<<grid_size, block_size>>>(a_input, a_info, bias);

    return 0;
}



}