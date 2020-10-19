#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_util.h"

#include <iostream>

__global__ void gpu_absval_forward_inplace(float* d_input, const int input_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x; //limited to 1024 rows

    if (index >= input_size) return;

    d_input[index] = d_input[index] >= 0 ? d_input[index] : -d_input[index];
}

namespace ncnn {

int relu_absval_forward_inplace(float* d_input, const int input_size)
{
    const int thread_per_block = 512;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);

    gpu_absval_forward_inplace<<<grid_size, block_size>>>(d_input, input_size);

    return 0;
}
} // namespace ncnn
