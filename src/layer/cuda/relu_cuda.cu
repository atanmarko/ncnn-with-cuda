//
// Author: Marko Atanasievski
//
// Copyright (C) 2020 TANCOM SOFTWARE SOLUTIONS Ltd. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.



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
    int thread_per_block = (((input_size - 1) / 32) + 1) * 32;
    if (thread_per_block > 1024) thread_per_block = 1024;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size((input_size - 1) / thread_per_block + 1, 1, 1);

    gpu_relu_forward_inplace<<<grid_size, block_size>>>(d_input, input_size, slope);

    return 0;
}

int relu_cuda_forward_inplace_int8(int8_t * d_input, int input_size, float slope)
{
    const int thread_per_block = 512;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size((input_size - 1) / thread_per_block + 1, 1, 1);

    gpu_relu_forward_inplace_int8<<<grid_size, block_size>>>(d_input, input_size, slope);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return 0;
}

}