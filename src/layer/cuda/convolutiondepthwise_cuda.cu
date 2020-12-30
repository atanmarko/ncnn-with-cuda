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
// Parts of this file are originally copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.



#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_util.h"

#include <iostream>

#include "mat.h"

#include "convolutiondepthwise_cuda.h"


static __device__ inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

__global__ void gpu_convolutiondepthwise_cuda_forward(const float* a_input, const ncnn::CudaMatInfo a_info,
                                              const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                              const float* bias_data, const float* activation_params,
                                              float* output, const ncnn::CudaMatInfo output_info,
                                              const ncnn::ConvolutionDepthWise_cuda::ConvolutionDepthWise_info product_info,
                                              const int* gpu_space_offset) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int current_group = blockIdx.z * blockDim.z + threadIdx.z;
    const int input_channel = current_group;


    extern __shared__ float buffer[];
    float* shared_kptr = buffer;

    const int k_index = threadIdx.x;

    if (k_index < product_info.maxk)
    {
        const float* kptr = (const float*)weight_data + product_info.maxk * current_group;
        shared_kptr[k_index] = kptr[k_index];
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h || current_group >= a_info.c)
    {
        return;
    }



    float sum = 0.f;
    if (product_info.bias_term)
    {
        sum += bias_data[current_group];
    }


    const float* sptr = a_input + input_channel * a_info.cstep + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;

    for (int k = 0; k < product_info.maxk; k++)
    {
        const float val = sptr [gpu_space_offset[k]];
        const float w = shared_kptr[k];
        sum += val * w;
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
        const float MISH_THRESHOLD = 20;
        float x = sum, y;
        if (x > MISH_THRESHOLD)
            y = x;
        else if (x < -MISH_THRESHOLD)
            y = expf(x);
        else
            y = logf(expf(x) + 1);
        sum = static_cast<float>(x * tanh(y));
    }

    const int output_index = current_group * output_info.cstep + output_row * output_info.w + output_column;
    output[output_index] = sum;
}

__global__ void gpu_convolutiondepthwise_cuda_forward_group(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             const float* weight_data, const ncnn::CudaMatInfo weight_info,
                                             const float* bias_data, const float* activation_params,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::ConvolutionDepthWise_cuda::ConvolutionDepthWise_info product_info,
                                             const int* gpu_space_offset,
                                             const int channels_g,
                                             const int num_output_g) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int total_output = blockIdx.z * blockDim.z + threadIdx.z;
    const int current_num_output_g = total_output % num_output_g;
    const int current_group = total_output / num_output_g;



    extern __shared__ float buffer[];
    float* shared_kptr = buffer;

    const int k_index = threadIdx.x;

    if (k_index < product_info.maxk)
    {
        const float* weight_data_ptr = (const float*)weight_data + product_info.maxk * channels_g * num_output_g * current_group;
        const float* kptr = (const float*)weight_data_ptr + product_info.maxk * channels_g * current_num_output_g;
        for (int input_channel = 0; input_channel < channels_g; input_channel++)
        {
            shared_kptr[input_channel * product_info.maxk + k_index] = kptr[input_channel * product_info.maxk + k_index];
        }
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h ||
        current_num_output_g >= num_output_g || current_group >= product_info.group)
    {
        return;
    }

    float sum = 0.f;
    if (product_info.bias_term)
    {
        sum += bias_data[num_output_g * current_group + current_num_output_g];
    }

    for (int input_channel = 0; input_channel < channels_g; input_channel++)
    {
        const float* sptr = a_input + (channels_g * current_group + input_channel) * a_info.cstep
                            + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;

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
        const float MISH_THRESHOLD = 20;
        float x = sum, y;
        if (x > MISH_THRESHOLD)
            y = x;
        else if (x < -MISH_THRESHOLD)
            y = expf(x);
        else
            y = logf(expf(x) + 1);
        sum = static_cast<float>(x * tanh(y));
    }

    const int output_index = (current_group*num_output_g+current_num_output_g) * output_info.cstep +
                             output_row * output_info.w + output_column;
    output[output_index] = sum;

}


__global__ void gpu_convolutiondepthwise_cuda_forward_int8(const signed char* a_input, const ncnn::CudaMatInfo a_info,
                                                      const signed char* weight_data, const ncnn::CudaMatInfo weight_info,
                                                      const float* bias_data, const float* activation_params,
                                                      signed char* output, const ncnn::CudaMatInfo output_info,
                                                      const ncnn::ConvolutionDepthWise_cuda::ConvolutionDepthWise_info product_info,
                                                      const int* gpu_space_offset,
                                                      const float *gpu_weight_data_int8_scales,
                                                      const float *gpu_bottom_blob_int8_scales) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int current_group = blockIdx.z * blockDim.z + threadIdx.z;
    const int input_channel = current_group;


    extern __shared__ float buffer[];
    float* shared_kptr = buffer;

    const int k_index = threadIdx.x;

    if (k_index < product_info.maxk)
    {
        const signed char* kptr = (const signed char*)weight_data + product_info.maxk * current_group;
        shared_kptr[k_index] = kptr[k_index];
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h || current_group >= a_info.c)
    {
        return;
    }



    int sum = 0;

    const signed char* sptr = a_input + input_channel * a_info.cstep + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;

    for (int k = 0; k < product_info.maxk; k++)
    {
        const signed char val = sptr [gpu_space_offset[k]];
        const signed char w = shared_kptr[k];
        sum += val * w;
    }

    const int output_index = current_group * output_info.cstep + output_row * output_info.w + output_column;
    if (product_info.use_int8_requantize)
    {
        // requantize and relu
        float scale_in;
        if (gpu_weight_data_int8_scales[current_group] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (gpu_bottom_blob_int8_scales[current_group] * gpu_weight_data_int8_scales[current_group]);

        float sumfp32 = sum * scale_in;

        if (product_info.bias_term)
            sumfp32 += bias_data[current_group];

        float scale_out = *product_info.gpu_top_blob_int8_scale; //FIXME load param

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
        if (gpu_weight_data_int8_scales[current_group] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (gpu_bottom_blob_int8_scales[current_group] * gpu_weight_data_int8_scales[current_group]);

        float sumfp32 = sum * scale_in;

        if (product_info.bias_term)
            sumfp32 += bias_data[current_group];

        if (product_info.activation_type == 1)
        {
            sumfp32 = max(sumfp32, 0.f);
        }

        ((float*)output)[output_index] = sumfp32;
    }
}

__global__ void gpu_convolutiondepthwise_cuda_forward_group_int8(const signed char* a_input, const ncnn::CudaMatInfo a_info,
                                                            const signed char* weight_data, const ncnn::CudaMatInfo weight_info,
                                                            const float* bias_data, const float* activation_params,
                                                                 signed char* output, const ncnn::CudaMatInfo output_info,
                                                            const ncnn::ConvolutionDepthWise_cuda::ConvolutionDepthWise_info product_info,
                                                            const int* gpu_space_offset,
                                                            const int channels_g,
                                                            const int num_output_g,
                                                            const float *gpu_weight_data_int8_scales,
                                                            const float *gpu_bottom_blob_int8_scales) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int total_output = blockIdx.z * blockDim.z + threadIdx.z;
    const int current_num_output_g = total_output % num_output_g;
    const int current_group = total_output / num_output_g;



    extern __shared__ signed char buffer8[];
    signed char* shared_kptr = buffer8;

    const int k_index = threadIdx.x;

    if (k_index < product_info.maxk)
    {
        const signed char* weight_data_ptr = (const signed char*)weight_data + product_info.maxk * channels_g * num_output_g * current_group;
        const signed char* kptr = (const signed char*)weight_data_ptr + product_info.maxk * channels_g * current_num_output_g;
        for (int input_channel = 0; input_channel < channels_g; input_channel++)
        {
            shared_kptr[input_channel * product_info.maxk + k_index] = kptr[input_channel * product_info.maxk + k_index];
        }
    }

    __syncthreads();

    if (output_column >= output_info.w || output_row >= output_info.h ||
        current_num_output_g >= num_output_g || current_group >= product_info.group)
    {
        return;
    }

    int sum = 0;

    for (int input_channel = 0; input_channel < channels_g; input_channel++)
    {
        const signed char* sptr = a_input + (channels_g * current_group + input_channel) * a_info.cstep
                            + output_row * product_info.stride_h * a_info.w + output_column * product_info.stride_w;

        for (int k = 0; k < product_info.maxk; k++)
        {
            const signed char val = sptr [gpu_space_offset[k]];
            const signed char w = shared_kptr[input_channel * product_info.maxk + k];
            sum += val * w;
        }
    }

    const int output_index = (current_group * num_output_g + current_num_output_g) * output_info.cstep
                             + output_row * output_info.w + output_column;

    if (product_info.use_int8_requantize)
    {
        // requantize and relu
        float scale_in;
        if (gpu_weight_data_int8_scales[current_group] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (gpu_bottom_blob_int8_scales[current_group] * gpu_weight_data_int8_scales[current_group]);

        float sumfp32 = sum * scale_in;

        if (product_info.bias_term)
            sumfp32 += bias_data[current_group * num_output_g + current_num_output_g];

        float scale_out = *product_info.gpu_top_blob_int8_scale; //FIXME load param

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
        if (gpu_weight_data_int8_scales[current_group] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (gpu_bottom_blob_int8_scales[current_group] * gpu_weight_data_int8_scales[current_group]);

        float sumfp32 = sum * scale_in;

        if (product_info.bias_term)
            sumfp32 += bias_data[current_group * num_output_g + current_num_output_g];

        if (product_info.activation_type == 1)
        {
            sumfp32 =  max(sumfp32, 0.f);
        }

        ((float*)output)[output_index] = sumfp32;
    }

}

namespace ncnn {

int convolutiondepthwise_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const ConvolutionDepthWise_cuda::ConvolutionDepthWise_info& info)
{
    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    if (bottom_blob_info.c == info.group && info.group == info.num_output)
    {
        const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
        int thread_per_block_x = ((number_of_threads - 1) / 32 + 1) * 32;
        if (thread_per_block_x > 64) thread_per_block_x = 64;
        int thread_per_block_y = ((top_blob.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = info.group;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_convolutiondepthwise_cuda_forward<<<grid_size, block_size, info.group * info.maxk * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                                                   bottom_blob_info,
                                                                                                                   static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                                                                                   weight_info,
                                                                                                                   static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                                   static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                                   static_cast<float*>(top_blob.get_raw_data()),
                                                                                                                   top_blob_info,
                                                                                                                   info,
                                                                                                                   info.gpu_space_ofs.get());
    }
    else
    {
        const int channels_g = bottom_blob_info.c / info.group;
        const int num_output_g = info.num_output / info.group;

        const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
        int thread_per_block_x = ((number_of_threads - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((top_blob.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = num_output_g * info.group;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        gpu_convolutiondepthwise_cuda_forward_group<<<grid_size, block_size, num_output_g * channels_g * info.maxk * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                                                                        bottom_blob_info,
                                                                                                                                        static_cast<const float*>(info.gpu_weight_data->get_craw_data()),
                                                                                                                                        weight_info,
                                                                                                                                        static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                                                        static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                                                        static_cast<float*>(top_blob.get_raw_data()),
                                                                                                                                        top_blob_info,
                                                                                                                                        info,
                                                                                                                                        info.gpu_space_ofs.get(),
                                                                                                                                        channels_g,
                                                                                                                                        num_output_g);
    }

    return 0;
}

int convolutiondepthwise_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const ConvolutionDepthWise_cuda::ConvolutionDepthWise_info& info)
{
    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};
    const ncnn::CudaMatInfo weight_info{*info.gpu_weight_data};

    if (bottom_blob_info.c == info.group && info.group == info.num_output)
    {
        const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
        int thread_per_block_x = ((number_of_threads - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((top_blob.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = info.group;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_convolutiondepthwise_cuda_forward_int8<<<grid_size, block_size, info.group * info.maxk * sizeof(signed char)>>>(static_cast<const signed char*>(bottom_blob.get_craw_data()),
                                                                                                                              bottom_blob_info,
                                                                                                                              static_cast<const signed char*>(info.gpu_weight_data->get_craw_data()),
                                                                                                                              weight_info,
                                                                                                                              static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                                              static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                                              static_cast<signed char*>(top_blob.get_raw_data()),
                                                                                                                              top_blob_info,
                                                                                                                              info,
                                                                                                                              info.gpu_space_ofs.get(),
                                                                                                                              static_cast<const float*>(info.gpu_weight_data_int8_scales->get_craw_data()),
                                                                                                                              static_cast<const float*>(info.gpu_bottom_blob_int8_scales->get_craw_data()));
    }
    else
    {
        const int channels_g = bottom_blob_info.c / info.group;
        const int num_output_g = info.num_output / info.group;

        const int number_of_threads = top_blob.w > info.maxk ? top_blob.w : info.maxk;
        int thread_per_block_x = ((number_of_threads - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((top_blob.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = num_output_g * info.group;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        gpu_convolutiondepthwise_cuda_forward_group_int8<<<grid_size, block_size, num_output_g * channels_g * info.maxk * sizeof(signed char)>>>(static_cast<const signed char*>(bottom_blob.get_craw_data()),
                                                                                                                                                bottom_blob_info,
                                                                                                                                                static_cast<const signed char*>(info.gpu_weight_data->get_craw_data()),
                                                                                                                                                weight_info,
                                                                                                                                                static_cast<const float*>(info.gpu_bias_data->get_craw_data()),
                                                                                                                                                static_cast<const float*>(info.gpu_activation_params->get_craw_data()),
                                                                                                                                                static_cast<signed char*>(top_blob.get_raw_data()),
                                                                                                                                                top_blob_info,
                                                                                                                                                info,
                                                                                                                                                info.gpu_space_ofs.get(),
                                                                                                                                                channels_g,
                                                                                                                                                num_output_g,
                                                                                                                                                static_cast<const float*>(info.gpu_weight_data_int8_scales->get_craw_data()),
                                                                                                                                                static_cast<const float*>(info.gpu_bottom_blob_int8_scales->get_craw_data()));
    }

    return 0;
}

}
