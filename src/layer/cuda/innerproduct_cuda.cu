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
//



#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_util.h"

#include <iostream>

#include "mat.h"

#include "innerproduct_cuda.h"



__global__ void gpu_innerproduct_cuda_forward(const float* a_input, const ncnn::CudaMatInfo a_info,
                                              const float* weight_input, const ncnn::CudaMatInfo weight_info,
                                              const float* bias_input,
                                              float* output, const ncnn::CudaMatInfo output_info,
                                              const ncnn::InnerProduct_cuda::InnerProduct_info product_info,
                                              float* scratchpad_memory) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int total_channel = (blockIdx.z * blockDim.z + threadIdx.z);

    extern __shared__ float buffer[];
    const int num_output = total_channel / a_info.c;
    const int channel = total_channel % a_info.c;

    const int shared_col = threadIdx.x;
    const int shared_row = threadIdx.y;
    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    const int shared_mem_index = shared_row * blockWidth + shared_col;
    buffer[shared_mem_index] = 0;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c || num_output >= product_info.num_output)
    {
        return;
    }

    const int index = channel * a_info.cstep + row * a_info.w + column;
    buffer[shared_mem_index] = a_input[index];

    const float* weight_ptr = weight_input + num_output * a_info.w * a_info.h * a_info.c + a_info.w * a_info.h * channel + row * a_info.w + column;
    float temp = buffer[shared_mem_index];
    buffer[shared_mem_index] = buffer[shared_mem_index] * (*weight_ptr);

    __syncthreads();

    const int reduction_width = blockWidth;
    for (int i = (reduction_width + 1) / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {
            buffer[shared_mem_index] = buffer[shared_mem_index] + buffer[shared_mem_index + i];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        float block_sum = 0;
        //Add bias per output number
        if (channel == 0 && row == 0 && column == 0 && product_info.bias_term) {
            block_sum = bias_input[num_output];
        }
        const int height = a_info.h < blockHeight ? a_info.h : blockHeight;
        for (int i = 0; i < height; ++i)
        {
            block_sum += buffer[i * blockWidth];
        }

        const int blocks_per_output = gridDim.x * gridDim.y * a_info.c;
        const int block_index = blocks_per_output * num_output + channel * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
        scratchpad_memory[block_index] = block_sum;
    }


}


__global__ void gpu_innerproduct_cuda_forward_sum(float* output, const ncnn::CudaMatInfo output_info,
                                                  const float* activation_params,
                                                  const ncnn::InnerProduct_cuda::InnerProduct_info product_info,
                                                  const int blocks_per_output,
                                                  float* scratchpad_memory) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_output = column;

    if (column >= product_info.num_output) return;

    float sum = 0;
    for (int i=column*blocks_per_output; i<column*(blocks_per_output)+blocks_per_output; i++) {
        sum = sum + scratchpad_memory[i];
    }



    if (product_info.activation_type == 1)
    {
        sum = max(sum, 0.f);
    }
    else if (product_info.activation_type == 2)
    {
        float slope = activation_params[0];
        sum = sum > 0.f ? sum : sum * slope;
    }
    else if (product_info.activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (sum < min)
            sum = min;
        if (sum > max)
            sum = max;
    }
    else if (product_info.activation_type == 4)
    {
        sum = static_cast<float>(1.f / (1.f + exp(-sum)));
    }
    else if (product_info.activation_type == 5)
    {
        sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
    }

    output[num_output] = sum;
}



namespace ncnn {

int innerproduct_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const InnerProduct_cuda::InnerProduct_info& info,
                              float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size)
{

    checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));

    int thread_per_block_x = ((bottom_blob.w - 1) / 64 + 1) * 64;
    if (thread_per_block_x > 128) thread_per_block_x = 128;
    int thread_per_block_y = ((bottom_blob.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    const int thread_per_block_z = 1;
    const int total_number_of_channels = bottom_blob.c * info.num_output;
    const int total_number_of_columns = bottom_blob.w;
    const int total_number_of_rows = bottom_blob.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    gpu_innerproduct_cuda_forward<<<grid_size, block_size, 1000 * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                     bottom_blob_info,
                                                                                     static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                                                     weight_info,
                                                                                     static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                     static_cast<float*>(top_blob.get_raw_data()),
                                                                                     top_blob_info,
                                                                                     info,
                                                                                     gpu_scratchpad_memory);


    const int blocks_per_output = grid_size.x * grid_size.y * bottom_blob_info.c;
    const dim3 block_size_sum(thread_per_block_x, 1, 1);
    const dim3 grid_size_sum((info.num_output - 1) / thread_per_block_x + 1,
                         1,
                         1);

    gpu_innerproduct_cuda_forward_sum<<<grid_size_sum, block_size_sum>>>(static_cast<float*>(top_blob.get_raw_data()),
                                                                           top_blob_info,
                                                                           static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                           info,
                                                                           blocks_per_output,
                                                                           gpu_scratchpad_memory);


    return 0;
}

int innerproduct_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const InnerProduct_cuda::InnerProduct_info& info,
                                   float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size)
{
    return 0;
}

}