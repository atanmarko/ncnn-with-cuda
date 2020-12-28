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

__global__ void gpu_batchnorm_load_model(int channels, float eps, float* a_data_gpu, float* b_data_gpu,
                                         float* bias_data_gpu, float* slope_data_gpu, float* mean_data_gpu, float* var_data_gpu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    const float sqrt_var = static_cast<float>(sqrt(var_data_gpu[i] + eps));
    a_data_gpu[i] = bias_data_gpu[i] - slope_data_gpu[i] * mean_data_gpu[i] / sqrt_var;
    b_data_gpu[i] = slope_data_gpu[i] / sqrt_var;
}

// input is 1 dimension
__global__ void gpu_batchnorm_forward_inplace_1(float* d_input, const float* b_data, const float* a_data, const ncnn::CudaMatInfo mat_info, const int input_size)
{

    const int i = blockIdx.x * blockDim.x + threadIdx.x; //limited to 1024 rows
    if (i >= input_size) return;

    d_input[i] =  b_data[i] * d_input[i] + a_data[i];
}

__global__ void gpu_batchnorm_forward_inplace_2(float* d_input, const float* b_data, const float* a_data, const ncnn::CudaMatInfo mat_info, const int input_size)
{
    if (blockIdx.x * blockDim.x + threadIdx.x >= input_size) return;

    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / mat_info.w;
    const int column = (blockIdx.x * blockDim.x + threadIdx.x) % mat_info.w;

    float* ptr = (float*)((unsigned char*)d_input + mat_info.w * mat_info.elemsize * row);

    ptr[column] = b_data[row]*ptr[column]+a_data[row];

}

__global__ void gpu_batchnorm_forward_inplace_3(float* d_input, const float* b_data, const float* a_data, const ncnn::CudaMatInfo mat_info, const int input_size)
{
    if (blockIdx.x * blockDim.x + threadIdx.x >= input_size) return;

    const int channelSize = mat_info.cstep;
    const int channel = (blockIdx.x * blockDim.x + threadIdx.x) / channelSize;
    const int row = ((blockIdx.x * blockDim.x + threadIdx.x) - (channel*channelSize)) / mat_info.w;
    const int column = ((blockIdx.x * blockDim.x + threadIdx.x) - (channel*channelSize)) % mat_info.w;


    const int step = channel * mat_info.cstep * mat_info.elemsize;
    float* ptr = (float*)((unsigned char*)d_input + step);

    const int i = row * mat_info.w+column;
    ptr[i] = b_data[channel] * ptr[i] + a_data[channel];
}

namespace ncnn {


int batchnorm_cuda_load_model(int channels, float eps, float* a_data_gpu, float* b_data_gpu,
                         float* bias_data_gpu, float* slope_data_gpu, float* mean_data_gpu, float* var_data_gpu)
{

    gpu_batchnorm_load_model<<<1, channels>>>(channels, eps, a_data_gpu, b_data_gpu,
                                                bias_data_gpu, slope_data_gpu, mean_data_gpu,
                                                var_data_gpu);

    return 0;
}




int batchnorm_cuda_forward_inplace(float* d_input, const float* b_data, const float* a_data, const CudaMatInfo& matInfo)
{


    if (matInfo.dims == 1)
    {
        const int input_size = matInfo.w;
        int thread_per_block = (((input_size - 1) / 32) + 1) * 32;
        if (thread_per_block > 1024) thread_per_block = 1024;
        dim3 block_size(thread_per_block,1,1);
        dim3 grid_size((matInfo.w - 1) / thread_per_block + 1, 1, 1);
        gpu_batchnorm_forward_inplace_1<<<grid_size, block_size>>>(d_input, b_data, a_data, matInfo, input_size);
    }
    if (matInfo.dims == 2)
    {
        const int input_size = matInfo.w * matInfo.h;
        int thread_per_block = (((input_size - 1) / 32) + 1) * 32;
        if (thread_per_block > 1024) thread_per_block = 1024;
        dim3 block_size(thread_per_block,1,1);
        dim3 grid_size( (input_size - 1) / thread_per_block + 1, 1, 1);
        gpu_batchnorm_forward_inplace_2<<<grid_size, block_size>>>(d_input, b_data, a_data, matInfo, input_size);
    }
    if (matInfo.dims == 3)
    {
        const int total_input_size = matInfo.cstep * matInfo.c;
        int thread_per_block = (((total_input_size - 1) / 32) + 1) * 32;
        if (thread_per_block > 1024) thread_per_block = 1024;
        dim3 block_size(thread_per_block,1,1);
        dim3 grid_size((total_input_size - 1) / thread_per_block + 1, 1, 1);
        gpu_batchnorm_forward_inplace_3<<<grid_size, block_size>>>(d_input, b_data, a_data, matInfo, total_input_size);
    }

    return 0;
}

} // namespace ncnn
