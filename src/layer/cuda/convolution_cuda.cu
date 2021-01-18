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

#include "convolution_cuda.h"


static __device__ inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

__global__ void gpu_convolution_cuda_forward(const float* a_input, const ncnn::CudaMatInfo a_info,
                                              const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                              const float* bias_data, const float* activation_params,
                                              float* output, const ncnn::CudaMatInfo output_info,
                                              const ncnn::Convolution_cuda::Convolution_info product_info,
                                              const int* const gpu_space_offset) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = blockIdx.z * blockDim.z + threadIdx.z;


    extern __shared__ float buffer[];
    float* shared_kptr = buffer;

    const int k_index = threadIdx.x;

    if (k_index < product_info.maxk)
    {
        const float* kptr = (const float*)weight_data + product_info.maxk * a_info.c * num_output;
        for (int input_channel = 0; input_channel < a_info.c; input_channel++)
        {
            shared_kptr[input_channel * product_info.maxk + k_index] = kptr[input_channel * product_info.maxk + k_index];
        }
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c)
    {
        return;
    }

    float sum = 0.f;
    if (product_info.bias_term)
    {
        sum += bias_data[num_output];
    }

    for (int input_channel = 0; input_channel < a_info.c; input_channel++)
    {
        const float* sptr = a_input + input_channel * a_info.cstep + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;

        for (int k = 0; k < product_info.maxk; k++)
        {
            const float val = sptr [gpu_space_offset[k]];
            const float w = shared_kptr[input_channel * product_info.maxk + k];
            sum += val * w;
        }
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

    const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;
    output[output_index] = sum;

}

__global__ void gpu_convolution_cuda_forward_02(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                             const float* bias_data, const float* activation_params,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Convolution_cuda::Convolution_info product_info,
                                             const int* const gpu_space_offset) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = (blockIdx.z * blockDim.z + threadIdx.z)/a_info.c;
    const int input_channel = (blockIdx.z * blockDim.z + threadIdx.z)%a_info.c;

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c || input_channel >= a_info.c)
    {
        return;
    }

    float partial_sum = 0.f;
    if (input_channel == 0 && product_info.bias_term)
    {
        partial_sum += bias_data[num_output];
    }

    const float* kptr = (const float*)weight_data + product_info.maxk * a_info.c * num_output;
    const float* sptr = a_input + input_channel * a_info.cstep + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;
    for (int k = 0; k < product_info.maxk; k++)
    {
        const float val = sptr[gpu_space_offset[k]];
        const float w = kptr[input_channel * product_info.maxk + k];
        partial_sum += val * w;
    }

    const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;
    atomicAdd(static_cast<float*>(output+output_index), partial_sum);
}

__global__ void gpu_convolution_cuda_forward_02_sum(const float* activation_params,
                                                    float* output,
                                                    const ncnn::CudaMatInfo output_info,
                                                    const ncnn::Convolution_cuda::Convolution_info convolution_info)
{
    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = (blockIdx.z * blockDim.z + threadIdx.z);

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c)
    {
        return;
    }

    const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;
    float sum = output[output_index];

    if (convolution_info.activation_type == 1)
    {
        sum = max(sum, 0.f);
    }
    else if (convolution_info.activation_type == 2)
    {
        float slope = activation_params[0];
        sum = sum > 0.f ? sum : sum * slope;
    }
    else if (convolution_info.activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (sum < min)
            sum = min;
        if (sum > max)
            sum = max;
    }
    else if (convolution_info.activation_type == 4)
    {
        sum = static_cast<float>(1.f / (1.f + exp(-sum)));
    }
    else if (convolution_info.activation_type == 5)
    {
        sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
    }

    output[output_index] = sum;
}


__global__ void gpu_convolution_cuda_forward_int8(const signed char* a_input, const ncnn::CudaMatInfo a_info,
                                             const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                             const float* bias_data, const float* activation_params,
                                             signed char* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Convolution_cuda::Convolution_info product_info,
                                             const int* const gpu_space_offset,
                                             const float *gpu_weight_data_int8_scales) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = blockIdx.z * blockDim.z + threadIdx.z;


    extern __shared__ signed char buffer_int8[];
    signed char* shared_kptr = buffer_int8;

    const int k_index = threadIdx.x;

    if (k_index < product_info.maxk)
    {
        const signed char* kptr = (const signed char*)weight_data + product_info.maxk * a_info.c * num_output;
        for (int input_channel = 0; input_channel < a_info.c; input_channel++)
        {
            shared_kptr[input_channel * product_info.maxk + k_index] = kptr[input_channel * product_info.maxk + k_index];
        }
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c)
    {
        return;
    }

    int sum = 0;

    for (int input_channel = 0; input_channel < a_info.c; input_channel++)
    {
        const signed char* sptr = a_input + input_channel * a_info.cstep + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;

        for (int k = 0; k < product_info.maxk; k++)
        {
            const int val = sptr [gpu_space_offset[k]];
            const int w = shared_kptr[input_channel * product_info.maxk + k];
            sum += val * w;
        }
    }

    const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;

    if (product_info.use_int8_requantize)
    {
        // requantize and relu
        float scale_in;
        if (gpu_weight_data_int8_scales[num_output] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (*product_info.gpu_bottom_blob_int8_scale * gpu_weight_data_int8_scales[num_output]);

        float sumfp32 = sum * scale_in;

        if (product_info.bias_term)
            sumfp32 += bias_data[num_output];

        float scale_out = *product_info.gpu_top_blob_int8_scale;

        signed char sums8 = float2int8(sumfp32 * scale_out);

        if (product_info.activation_type == 1)
        {
            sums8 = max(sums8, (signed char)0);
        }

        output[output_index] = sums8;
    }
    else
    {
        // dequantize and relu
        float scale_in;
        if (gpu_weight_data_int8_scales[num_output] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (*product_info.gpu_bottom_blob_int8_scale * gpu_weight_data_int8_scales[num_output]);

        float sumfp32 = sum * scale_in;
        if (product_info.bias_term)
            sumfp32 += bias_data[num_output];

        if (product_info.activation_type == 1)
        {
            sumfp32 = max(sumfp32, 0.f);
        }

        ((float*)output)[output_index] = sumfp32;
    }



}




__global__ void gpu_convolution_cuda_forward_03(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                             const float* bias_data, const float* activation_params,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Convolution_cuda::Convolution_info product_info,
                                             const int* const gpu_space_offset)
{
    const int input_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int input_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_channel = blockIdx.z * blockDim.z + threadIdx.z;

//    const int block_idx = blockIdx.z* gridDim.x*gridDim.y+blockIdx.y*blockDim.x+blockIdx.x;

    extern __shared__ float buffer[];
    float* shared_input = buffer;
    float* shared_kptr = buffer + (blockDim.z * blockDim.x * blockDim.y) * product_info.maxk;
    float* shared_partial_sums = buffer + (blockDim.z * blockDim.x * blockDim.y  * product_info.maxk) + (blockDim.z * product_info.maxk);

    const int buffer_column = threadIdx.x;
    const int buffer_row = threadIdx.y;
    const int buffer_channel = threadIdx.z;

    const int shared_partial_sums_index = buffer_channel * blockDim.x * blockDim.y + buffer_row * blockDim.x + buffer_column;
    shared_partial_sums[shared_partial_sums_index] = 0;

    if (input_column >= a_info.w || input_row >= a_info.h || input_channel >= a_info.c)
    {
        return;
    }


    const int output_row = input_row / product_info.stride_h;
    const int output_column = input_column / product_info.stride_w;

    for (int k_index=0; k_index<product_info.maxk; ++k_index)
    {
        const int input_index = input_channel * a_info.cstep + input_row * a_info.w + input_column + gpu_space_offset[k_index];
        const int buffer_index = buffer_channel * product_info.maxk * blockDim.x * blockDim.y + buffer_row * blockDim.x * product_info.maxk + buffer_column * product_info.maxk + k_index;
        shared_input[buffer_index] = a_input[input_index];

        //        if ((input_row >=0 && input_row<=1) && input_channel == 0 && (input_column == 3 || input_column == 4)) {
//            printf("GPU input: input_channel: %d  input_row: %d input_column: %d block_idx: %d output_row: %d output_column: %d buffer_channel: %d buffer_row: %d buffer_column: %d k_index:%d input_index: %d buffer_index: %d gpu_space_offset[k_index]: %d value: %f\n ",
//                   input_channel, input_row, input_column, block_idx, output_row, output_column, buffer_channel, buffer_row, buffer_column, k_index, input_index, buffer_index, gpu_space_offset[k_index], a_input[input_index]);
//        }
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h)
        return;

    if ((input_row % product_info.stride_h != 0) || (input_column % product_info.stride_w != 0))
        return;


    for (int num_output = 0; num_output < product_info.num_output; ++num_output)
    {
        //load mask
        const float* kptr = (const float*)weight_data + product_info.maxk * a_info.c * num_output;
//        if (buffer_row == 0 && buffer_column == 0)
        {
            for (int k2 = 0; k2 < product_info.maxk; k2++)
            {
                shared_kptr[buffer_channel * product_info.maxk + k2] = kptr[input_channel * product_info.maxk + k2];

//                if (num_output == 0 && (input_channel >= 0 && input_channel <= 0))
//                    printf("GPU KPTR block_idx: %d kernel_w: %d kernel_h: %d input_channel:%d buffer_channel: %d product_info.maxk: %d buffer_row: %d buffer_column:%d value: %f k:%d\n",
//                           block_idx, product_info.kernel_w, product_info.kernel_h, input_channel, buffer_channel, product_info.maxk, buffer_row, buffer_column, shared_kptr[buffer_channel * product_info.maxk + k2], k2);
            }
        }

        __syncthreads();

        float partial_sum = 0.f;
        if (buffer_channel == 0 && product_info.bias_term)
        {
            partial_sum += bias_data[num_output];
        }

        for (int k = 0; k < product_info.maxk; k++)
        {
            const float val = shared_input[buffer_channel * product_info.maxk * blockDim.x * blockDim.y + buffer_row * blockDim.x * product_info.maxk + buffer_column * product_info.maxk + k];
            const float w = shared_kptr[buffer_channel * product_info.maxk + k];
            partial_sum += val * w;
//            if (num_output == 0 && output_row == 0 && output_column == 2 && (input_channel >= 0 && input_channel <= 0))
//                printf("GPU block_idx: %d stride_w: %d stride_h: %d buffer_channel: %d input channel: %d input_row: %d input_column: %d num_output: %d output_row: %d output_column: %d maxk: %d k: %d buffer index: %d val: %f w: %f partial_sum: %f\n",
//                       block_idx, product_info.stride_w, product_info.stride_h, buffer_channel, input_channel,
//                       input_row, input_column, num_output, output_row, output_column, product_info.maxk, k, buffer_channel * product_info.maxk + k, val, w, partial_sum);
        }
        shared_partial_sums[shared_partial_sums_index] = partial_sum;

        __syncthreads();

        if (buffer_channel == 0)
        {
            float num_output_block_sum = 0.f;
            const int min_z = a_info.c < blockDim.z ? a_info.c : blockDim.z;
            for (int i = 0; i <  min_z; ++i)
            {
                const int current_shared_partial_sums_index = i * blockDim.x * blockDim.y + buffer_row * blockDim.x + buffer_column;
                num_output_block_sum += shared_partial_sums[current_shared_partial_sums_index];
            }


            const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;
            atomicAdd(static_cast<float*>(output + output_index), num_output_block_sum);
        }
        __syncthreads();
    }
}

__global__ void gpu_convolution_cuda_forward_03_sum(const float* activation_params,
                                                    float* output,
                                                    const ncnn::CudaMatInfo output_info,
                                                    const ncnn::Convolution_cuda::Convolution_info convolution_info)
{
    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = (blockIdx.z * blockDim.z + threadIdx.z);

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c)
    {
        return;
    }

    const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;
    float sum = output[output_index];

    if (convolution_info.activation_type == 1)
    {
        sum = max(sum, 0.f);
    }
    else if (convolution_info.activation_type == 2)
    {
        float slope = activation_params[0];
        sum = sum > 0.f ? sum : sum * slope;
    }
    else if (convolution_info.activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (sum < min)
            sum = min;
        if (sum > max)
            sum = max;
    }
    else if (convolution_info.activation_type == 4)
    {
        sum = static_cast<float>(1.f / (1.f + exp(-sum)));
    }
    else if (convolution_info.activation_type == 5)
    {
        sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
    }

    output[output_index] = sum;
}


__global__ void gpu_convolution_cuda_transform(const float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    const int input_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int input_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int input_channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (input_column >= a_info.w || input_row >= a_info.h || input_channel >= a_info.c)
    {
        return;
    }

    int input_index = input_channel * a_info.cstep + input_row * a_info.w + input_column;
    int output_index = input_row*a_info.w*a_info.c + input_column*a_info.c + input_channel;
    scratchpad_memory[output_index] = a_input[input_index];
}





__global__ void gpu_convolution_cuda_forward_04(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                             const float* bias_data, const float* activation_params,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Convolution_cuda::Convolution_info convolution_info,
                                             const int* const gpu_space_offset,
                                             float* scratchpad_memory) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = (blockIdx.z * blockDim.z + threadIdx.z) % convolution_info.num_output;
    const int k_index = (blockIdx.z * blockDim.z + threadIdx.z) / convolution_info.num_output;

//    const int block_idx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * blockDim.x + blockIdx.x;

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c || k_index >= convolution_info.maxk)
    {
        return;
    }

    float partial_sum = 0.f;
    if (convolution_info.bias_term && k_index == 0)
    {
        partial_sum += bias_data[num_output];
    }

    const float* kptr = (const float*)weight_data + convolution_info.maxk * a_info.c * num_output;

    const float* sptr = a_input + output_row * convolution_info.stride_h * a_info.w * a_info.c + output_column * convolution_info.stride_w * a_info.c
                        + gpu_space_offset[k_index] * a_info.c;

    for (int input_channel = 0; input_channel < a_info.c; input_channel++)
    {
        const float val = sptr[input_channel];
        const float w = kptr[input_channel * convolution_info.maxk + k_index];
        partial_sum += val * w;
    }

    const int scratchpad_index = (a_info.c*a_info.w*a_info.h)+(num_output * output_info.w * output_info.h + output_row * output_info.w + output_column)*convolution_info.maxk+ k_index;
    scratchpad_memory[scratchpad_index] = partial_sum;
}


__global__ void gpu_convolution_cuda_forward_04_sum(const float* activation_params,
                                                    float* output,
                                                    const ncnn::CudaMatInfo input_info,
                                                    const ncnn::CudaMatInfo output_info,
                                                    const ncnn::Convolution_cuda::Convolution_info convolution_info,
                                                    float* scratchpad_memory)
{
    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_output = (blockIdx.z * blockDim.z + threadIdx.z);

    if (output_column >= output_info.w || output_row >= output_info.h || num_output >= output_info.c)
    {
        return;
    }

    const int output_index = num_output * output_info.cstep + output_row * output_info.w + output_column;
    float sum = 0;

    const int scratchpad_index = (input_info.c * input_info.w * input_info.h) + (num_output * output_info.w * output_info.h
                                                                                 + output_row * output_info.w + output_column) * convolution_info.maxk;


    for (int i = 0; i < convolution_info.maxk; ++i)
    {
        sum += scratchpad_memory[scratchpad_index + i];
    }

    if (convolution_info.activation_type == 1)
    {
        sum = max(sum, 0.f);
    }
    else if (convolution_info.activation_type == 2)
    {
        float slope = activation_params[0];
        sum = sum > 0.f ? sum : sum * slope;
    }
    else if (convolution_info.activation_type == 3)
    {
        float min = activation_params[0];
        float max = activation_params[1];
        if (sum < min)
            sum = min;
        if (sum > max)
            sum = max;
    }
    else if (convolution_info.activation_type == 4)
    {
        sum = static_cast<float>(1.f / (1.f + exp(-sum)));
    }
    else if (convolution_info.activation_type == 5)
    {
        sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
    }

    output[output_index] = sum;
}




namespace ncnn {


int convolution_cuda_forward_04(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info,
                                float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size)
{
    //transform input

    if ((bottom_blob.total() + top_blob.total()*info.maxk)*sizeof(float) > gpu_scratchpad_memory_size) {
        std::cout << "CONVOLUTION current scratchpad memory: " << gpu_scratchpad_memory_size << " required: "
                  << (bottom_blob.total() + top_blob.total() * info.maxk) * sizeof(float) << std::endl;
        throw  std::runtime_error("Not enough scratchpad memory");
    }

    int thread_per_block_transform_x = ((bottom_blob.w - 1) / 16 + 1) * 16;
    if (thread_per_block_transform_x > 16) thread_per_block_transform_x = 16;
    int thread_per_block_transform_y = ((bottom_blob.h - 1) / 2 + 1) * 2;
    if (thread_per_block_transform_y > 2) thread_per_block_transform_y = 2;
    const int thread_per_block_transform_z = 16;
    const int total_number_of_columns_transform = bottom_blob.w;
    const int total_number_of_rows_transform = bottom_blob.h;
    const int total_number_of_channels_transform = bottom_blob.c;

    const dim3 block_size_transform(thread_per_block_transform_x, thread_per_block_transform_y, thread_per_block_transform_z);
    const dim3 grid_size_transform((total_number_of_columns_transform - 1) / thread_per_block_transform_x + 1,
                         (total_number_of_rows_transform - 1) / thread_per_block_transform_y + 1,
                         (total_number_of_channels_transform - 1) / thread_per_block_transform_z + 1);

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    gpu_convolution_cuda_transform<<<grid_size_transform, block_size_transform>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                  bottom_blob_info,
                                                                                  gpu_scratchpad_memory);

    const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
    int thread_per_block_x = ((number_of_threads - 1) / 8 + 1) * 8;
    if (thread_per_block_x > 8) thread_per_block_x = 8;
    int thread_per_block_y = ((top_blob.h - 1) / 2 + 1) * 2;
    if (thread_per_block_y > 2) thread_per_block_y = 2;
    int thread_per_block_z = ((top_blob.c - 1) / 32 + 1) * 32;
    if (thread_per_block_z > 64) thread_per_block_z = 64;

    const int total_number_of_columns = top_blob.w;
    const int total_number_of_rows = top_blob.h;
    const int total_number_of_channels = top_blob.c * info.maxk;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    gpu_convolution_cuda_forward_04<<<grid_size, block_size>>>(static_cast<const float*>(gpu_scratchpad_memory),
                                                                                                       bottom_blob_info,
                                                                                                       static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                                                                       weight_info,
                                                                                                       static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                       static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                       static_cast<float*>(top_blob.get_raw_data()),
                                                                                                       top_blob_info,
                                                                                                       info,
                                                                                                       static_cast<const int*>(info.gpu_space_ofs),
                                                                                                          gpu_scratchpad_memory);

    int thread_per_block_sum_x = ((top_blob.w - 1) / 16 + 1) * 16;
    if (thread_per_block_sum_x > 16) thread_per_block_sum_x = 16;
    int thread_per_block_sum_y = ((bottom_blob.h - 1) / 2 + 1) * 2;
    if (thread_per_block_sum_y > 2) thread_per_block_sum_y = 2;
    const int thread_per_block_sum_z = 16;
    const int total_number_of_columns_sum = top_blob.w;
    const int total_number_of_rows_sum = top_blob.h;
    const int total_number_of_channels_sum = top_blob.c;

    const dim3 block_size_sum(thread_per_block_sum_x, thread_per_block_sum_y, thread_per_block_sum_z);
    const dim3 grid_size_sum((total_number_of_columns_sum - 1) / thread_per_block_sum_x + 1,
                                   (total_number_of_rows_sum - 1) / thread_per_block_sum_y + 1,
                                   (total_number_of_channels_sum - 1) / thread_per_block_sum_z + 1);

    gpu_convolution_cuda_forward_04_sum<<<grid_size_sum, block_size_sum>>>(static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                             static_cast<float*>(top_blob.get_raw_data()),
                                                                             bottom_blob_info,
                                                                             top_blob_info,
                                                                             info,
                                                                             gpu_scratchpad_memory);

    return 0;
}

int convolution_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info)
{
    const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
    int thread_per_block_x = ((number_of_threads - 1) / 32 + 1) * 32;
    if (thread_per_block_x > 64) thread_per_block_x = 64;
    int thread_per_block_y = ((top_blob.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    const int thread_per_block_z = 1;
    const int total_number_of_channels = top_blob.c;
    const int total_number_of_columns = top_blob.w;
    const int total_number_of_rows = top_blob.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    gpu_convolution_cuda_forward<<<grid_size, block_size, bottom_blob.c * info.maxk * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                                       bottom_blob_info,
                                                                                                       static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                                                                       weight_info,
                                                                                                       static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                       static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                       static_cast<float*>(top_blob.get_raw_data()),
                                                                                                       top_blob_info,
                                                                                                       info,
                                                                                                       static_cast<const int*>(info.gpu_space_ofs));

    return 0;
}

int convolution_cuda_forward_03(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info)
{

    int temp = 32*1024/(info.maxk*sizeof(float));
    int thread_per_block_x = 3;
    int thread_per_block_y = 3;
    int thread_per_block_z = temp/(thread_per_block_x*thread_per_block_y);
    if (thread_per_block_z > 64) thread_per_block_z = 64;
    const int total_number_of_channels = bottom_blob.c;
    const int total_number_of_columns = bottom_blob.w;
    const int total_number_of_rows = bottom_blob.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    const int shared_mem_size =  ((thread_per_block_z *  thread_per_block_x * thread_per_block_y * (info.maxk)) + (info.maxk*thread_per_block_z)
                                 + thread_per_block_z * thread_per_block_x * thread_per_block_y) * sizeof(float);

//    static int counter = 0;
//    std::cout << "COUNTER: " << counter << " shared_mem_size: " << shared_mem_size << " grid_size: x:" << grid_size.x << " y: " << grid_size.y << " z:" << grid_size.z << " block_size: x: " << block_size.x << " y: " << block_size.y <<
//        " z:" << block_size.z << std::endl;
//    counter++;
//    std::cout << "Padding: left: " << info.pad_left << " right: " << info.pad_right << std::endl;

    gpu_convolution_cuda_forward_03<<<grid_size, block_size, shared_mem_size>>>
        (static_cast<const float*>(bottom_blob.get_craw_data()),
                                                            bottom_blob_info,
                                                            static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                            weight_info,
                                                            static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                            static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                            static_cast<float*>(top_blob.get_raw_data()),
                                                            top_blob_info,
                                                            info,
                                                            static_cast<const int*>(info.gpu_space_ofs));


    int thread_per_block_x_sum = ((top_blob.w - 1) / 32 + 1) * 32;
    if (thread_per_block_x_sum > 32) thread_per_block_x_sum = 32;
    int thread_per_block_y_sum = ((top_blob.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y_sum > 8) thread_per_block_y_sum = 8;
    const int thread_per_block_z_sum = 4;
    const int total_number_of_channels_sum = top_blob.c;
    const int total_number_of_columns_sum = top_blob.w;
    const int total_number_of_rows_sum = top_blob.h;

    const dim3 block_size_sum(thread_per_block_x_sum, thread_per_block_y_sum, thread_per_block_z_sum);
    const dim3 grid_size_sum((total_number_of_columns_sum - 1) / thread_per_block_x_sum + 1,
                         (total_number_of_rows_sum - 1) / thread_per_block_y_sum + 1,
                         (total_number_of_channels_sum - 1) / thread_per_block_z_sum + 1);

//    std::cout << "shared_mem_size: " << shared_mem_size << " grid_size_sum: x:" << grid_size_sum.x << " y: " << grid_size_sum.y << " z:" << grid_size_sum.z << " block_size_sum: x: "
//              << block_size_sum.x << " y: " << block_size_sum.y << " z:" << block_size_sum.z << std::endl;

    gpu_convolution_cuda_forward_03_sum<<<grid_size, block_size>>>(static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
        static_cast<float*>(top_blob.get_raw_data()),
        top_blob_info,
        info);

    return 0;
}

int convolution_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info)
{
    const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
    int thread_per_block_x = ((number_of_threads - 1) / 64 + 1) * 64;
    if (thread_per_block_x > 128) thread_per_block_x = 128;
    int thread_per_block_y = ((top_blob.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    const int thread_per_block_z = 1;
    const int total_number_of_channels = top_blob.c;
    const int total_number_of_columns = top_blob.w;
    const int total_number_of_rows = top_blob.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    gpu_convolution_cuda_forward_int8<<<grid_size, block_size, bottom_blob.c * info.maxk * sizeof(signed char)>>>(static_cast<const signed char*>(bottom_blob.get_craw_data()),
                                                                                                       bottom_blob_info,
                                                                                                       static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                                                                       weight_info,
                                                                                                       static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                       static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                       static_cast<signed char*>(top_blob.get_raw_data()),
                                                                                                       top_blob_info,
                                                                                                       info,
                                                                                                       static_cast<const int*>(info.gpu_space_ofs),
                                                                                                       static_cast<const float*>(info.gpu_weight_data_int8_scales->get_craw_data()));

    return 0;
}


}
