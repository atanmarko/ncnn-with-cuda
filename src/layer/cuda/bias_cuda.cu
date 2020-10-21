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