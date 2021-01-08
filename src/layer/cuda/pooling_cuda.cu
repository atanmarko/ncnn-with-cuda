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
#include <float.h>

#include "pooling_cuda.h"



__global__ void gpu_pooling_cuda_forward_global_max(const float* a_input, const ncnn::CudaMatInfo a_info,
                                              float* output, const ncnn::CudaMatInfo output_info,
                                              const ncnn::Pooling_cuda::Pooling_info pooling_info,
                                             int* pooling_lock) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    const int block_column = threadIdx.x;
    const int block_row = threadIdx.y;

    extern __shared__ float buffer[];

    buffer[block_row * blockDim.x + block_column] = -FLT_MAX;
    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    const float* input_ptr = a_input+channel*a_info.cstep + row*a_info.w + column;
    buffer[block_row * blockDim.x + block_column] = *input_ptr;
    __syncthreads();



    if (block_column == 0)
    {
        float max_row_value = -FLT_MAX;
        for (int i = 0; i < blockDim.x; ++i)
        {
            if (buffer[block_row * blockDim.x + i] > max_row_value)
            {
                max_row_value = buffer[block_row * blockDim.x + i];
            }
        }

        buffer[block_row * blockDim.x] = max_row_value;
        __syncthreads();

        if (block_row == 0) {
            float max_block_value = -FLT_MAX;

            for (int i=0; i<blockDim.y; ++i) {
                if (max_block_value < buffer[i*blockDim.x])
                {
                    max_block_value = buffer[i * blockDim.x];
                }
            }

            cuda_lock(pooling_lock);
            if (output[channel] < max_block_value)
            {
                output[channel] = max_block_value;
            }
            cuda_unlock(pooling_lock);
        }

    }
}

__global__ void gpu_pooling_cuda_forward_global_sum(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Pooling_cuda::Pooling_info pooling_info,
                                             int* pooling_lock) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    const int block_column = threadIdx.x;
    const int block_row = threadIdx.y;

    extern __shared__ float buffer[];

    buffer[block_row * blockDim.x + block_column] = 0;
    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    const float* input_ptr = a_input+channel*a_info.cstep + row*a_info.w + column;
    buffer[block_row * blockDim.x + block_column] = *input_ptr;
    __syncthreads();

    if (block_column == 0)
    {
        float sum_row_value = 0;
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum_row_value += buffer[block_row * blockDim.x + i];
        }

//        printf("Row: %d sum_row_value: %f\n", row, sum_row_value);

        buffer[block_row * blockDim.x] = sum_row_value;
        __syncthreads();

        if (block_row == 0) {
            float sum_block_value = 0;

            for (int i=0; i<blockDim.y; ++i) {
                sum_block_value += buffer[i*blockDim.x];
            }
//            printf("Row: %d  sum_block_value: %f\n", row,  sum_block_value);

            cuda_lock(pooling_lock);
            output[channel] += sum_block_value;
//            printf("TWO channel:%d output[channel]: %f sum_block_value: %f\n", channel, output[channel], sum_block_value);
            cuda_unlock(pooling_lock);
        }

    }
}

__global__ void gpu_pooling_cuda_forward_global_sum_total(float* output, const ncnn::CudaMatInfo input_info, const int size) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= 1 || row >= 1 || channel >= input_info.c)
    {
        return;
    }
    output[channel] = output[channel] / size;
}

__global__ void gpu_pooling_cuda_forward_max(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Pooling_cuda::Pooling_info pooling_info,
                                             const int* gpu_space_offset) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (output_column >= output_info.w || output_row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    const float* input_ptr = a_input + channel * a_info.cstep + output_row * a_info.w * pooling_info.stride_h + output_column * pooling_info.stride_w;

    float max_value = -FLT_MAX;

    for (int k = 0; k < pooling_info.maxk; k++)
    {
        const float val = input_ptr[gpu_space_offset[k]];
        max_value = max(max_value, val);
    }

    float* output_ptr = output + channel * output_info.cstep + output_row * output_info.w + output_column;
    *output_ptr = max_value;
}

__global__ void gpu_pooling_cuda_forward_sum0(const float* a_input, const ncnn::CudaMatInfo a_info,
                                                    float* output, const ncnn::CudaMatInfo output_info,
                                                    const ncnn::Pooling_cuda::Pooling_info pooling_info,
                                                    int* pooling_lock) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (output_column >= output_info.w || output_row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    const float* input_ptr_channel = a_input + channel * a_info.cstep;



    const int sy0 = output_row * pooling_info.stride_h;
    const int sx0 = output_column * pooling_info.stride_w;
    float sum = 0;
    int area = 0;

    for (int ki = 0; ki < pooling_info.kernel_h; ki++)
    {
        int sy = sy0 + ki;

        if (sy < pooling_info.pad_top)
            continue;

        if (sy >= a_info.h - pooling_info.pad_bottom - pooling_info.htailpad)
            break;

        for (int kj = 0; kj < pooling_info.kernel_w; kj++)
        {
            int sx = sx0 + kj;

            if (sx < pooling_info.pad_left)
                continue;

            if (sx >= a_info.w - pooling_info.pad_right - pooling_info.wtailpad)
                break;

            float val = *(input_ptr_channel+sy*a_info.w+sx);
            sum += val;
            area += 1;
        }


    }

    float* output_ptr = output + channel * output_info.cstep + output_row * output_info.w + output_column;
    *output_ptr = sum / area;
}


__global__ void gpu_pooling_cuda_forward_sum1(const float* a_input, const ncnn::CudaMatInfo a_info,
                                             float* output, const ncnn::CudaMatInfo output_info,
                                             const ncnn::Pooling_cuda::Pooling_info pooling_info,
                                             const int* gpu_space_offset) {

    const int output_column = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (output_column >= output_info.w || output_row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    const float* input_ptr = a_input + channel * a_info.cstep + output_row * a_info.w * pooling_info.stride_h + output_column * pooling_info.stride_w;

    float sum = 0;

    for (int k = 0; k < pooling_info.maxk; k++)
    {
        const float val = input_ptr[gpu_space_offset[k]];
        sum += val;
    }

    float* output_ptr = output + channel * output_info.cstep + output_row * output_info.w + output_column;
    *output_ptr = sum / pooling_info.maxk;
}


namespace ncnn {

int pooling_cuda_forward_global(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info)
{
    const int number_of_threads = bottom_blob.w;
    int thread_per_block_x = ((number_of_threads - 1) / 32 + 1) * 32;
    if (thread_per_block_x > 64) thread_per_block_x = 64;
    int thread_per_block_y = ((bottom_blob.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    const int thread_per_block_z = 1;
    const int total_number_of_channels = bottom_blob.c;
    const int total_number_of_columns = bottom_blob.w;
    const int total_number_of_rows = bottom_blob.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    const ncnn::CudaMatInfo top_blob_info{top_blob};

    if (info.pooling_type == Pooling::PoolMethod_MAX)
    {

        top_blob.fill(-FLT_MAX);
        gpu_pooling_cuda_forward_global_max<<<grid_size, block_size, block_size.x * block_size.y * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                                                      bottom_blob_info,
                                                                                                                      static_cast<float*>(top_blob.get_raw_data()),
                                                                                                                      top_blob_info,
                                                                                                                      info, info.pooling_lock);
    }
    else if (info.pooling_type == Pooling::PoolMethod_AVE)
    {
        top_blob.fill(0);
        gpu_pooling_cuda_forward_global_sum<<<grid_size, block_size, block_size.x * block_size.y * sizeof(float)>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                                                                      bottom_blob_info,
                                                                                                                      static_cast<float*>(top_blob.get_raw_data()),
                                                                                                                      top_blob_info,
                                                                                                                      info, info.pooling_lock);

        const int thread_per_block_z_sum = 16;

        const dim3 block_size_sum(1, 1, thread_per_block_z_sum);
        const dim3 grid_size_sum(1, 1, (total_number_of_channels - 1) / thread_per_block_z_sum + 1);

        gpu_pooling_cuda_forward_global_sum_total<<<grid_size_sum, block_size_sum>>>(static_cast<float*>(top_blob.get_raw_data()),
                                                                                       bottom_blob_info,
                                                                                       (bottom_blob_info.w * bottom_blob_info.h));
    }



    return 0;
}



int pooling_cuda_forward(const CudaMat& bottom_blob_bordered, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info)
{
    const int number_of_threads = top_blob.w;
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

    const ncnn::CudaMatInfo bottom_blob_bordered_info{bottom_blob_bordered};
    const ncnn::CudaMatInfo top_blob_info{top_blob};

    if (info.pooling_type == Pooling::PoolMethod_MAX)
    {

        top_blob.fill(-FLT_MAX);
        gpu_pooling_cuda_forward_max<<<grid_size, block_size>>>(static_cast<const float*>(bottom_blob_bordered.get_craw_data()),
                                                                  bottom_blob_bordered_info,
                                                                  static_cast<float*>(top_blob.get_raw_data()),
                                                                  top_blob_info,
                                                                  info,
                                                                  info.gpu_space_ofs.get());
    }
    else if (info.pooling_type == Pooling::PoolMethod_AVE)
    {
        if (info.avgpool_count_include_pad == 0)
        {
            top_blob.fill(0);
            gpu_pooling_cuda_forward_sum0<<<grid_size, block_size>>>(static_cast<const float*>(bottom_blob_bordered.get_craw_data()),
                                                                                                                    bottom_blob_bordered_info,
                                                                                                                    static_cast<float*>(top_blob.get_raw_data()),
                                                                                                                    top_blob_info,
                                                                                                                    info, info.pooling_lock);

        }
        else  {
            top_blob.fill(0);
            gpu_pooling_cuda_forward_sum1<<<grid_size, block_size>>>(static_cast<const float*>(bottom_blob_bordered.get_craw_data()),
                                                                    bottom_blob_bordered_info,
                                                                    static_cast<float*>(top_blob.get_raw_data()),
                                                                    top_blob_info,
                                                                    info,
                                                                    info.gpu_space_ofs.get());

        }

    }



    return 0;
}

}
