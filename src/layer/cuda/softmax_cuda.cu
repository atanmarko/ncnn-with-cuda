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
#include <nppdefs.h>


#include "cuda_util.h"
#include "mat.h"
#include "softmax_cuda.h"

#include <iostream>

__global__ void gpu_softmax_reduce_find_max_row(const float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    extern __shared__ float sh_buffer[]; //Shared working memory

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int index = channel * a_info.cstep + row * a_info.w + column;

    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    const int channel_shared_memory_step = blockHeight * blockWidth;

    float* max_value = sh_buffer;
    const int max_value_index = channel_shared_memory_step * threadIdx.z + threadIdx.y * blockWidth + threadIdx.x;
    max_value[max_value_index] = -NPP_MAXABS_32F;

    if (row >= a_info.h || column >= a_info.w || channel >= a_info.c) {
        return;
    }

    max_value[max_value_index] = a_input[index];
    __syncthreads();

    for (int i = (blockWidth+1) / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {
            if (max_value[max_value_index+i] > max_value[max_value_index]) {
                max_value[max_value_index] = max_value[max_value_index+i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        const int scratchpad_memory_offset_sum = a_info.h * a_info.c;
        const int scratchpad_channel_memory_step = gridDim.x * a_info.h;
        const int scracthpad_index = scratchpad_memory_offset_sum + channel * scratchpad_channel_memory_step + row * gridDim.x + blockIdx.x;
        scratchpad_memory[scracthpad_index] = max_value[max_value_index];

//        printf("CHECKPOINT MAX VALUE 1: channel: %d row: %d scratchpad_memory[channel_scratchpad_memory_step+row]: %f step: %d calculated: %f \n",
//               channel, row, scratchpad_memory[scracthpad_index], scracthpad_index, max_value[max_value_index]);
    }
}

__global__ void gpu_softmax_reduce_find_max_row_result( float* scratchpad_memory, dim3 result_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float sh_buffer[];
    float *final_result = sh_buffer;
    const int final_result_channel_step = result_size.y * threadIdx.z;
    final_result[final_result_channel_step + row] = -NPP_MAXABS_32F;

    if (column >= result_size.x || row >= result_size.y || channel >= result_size.z) return;

    const int scratchpad_memory_offset_sum = result_size.y * result_size.z;
    int channel_scratchpad_memory_step = result_size.x * result_size.y;
    const int scratchpad_max_value_index_step =  scratchpad_memory_offset_sum + channel * channel_scratchpad_memory_step + row * result_size.x;

    for (int i = 0; i < result_size.x; i++)
    {
        if (final_result[final_result_channel_step+row] < scratchpad_memory[scratchpad_max_value_index_step + i])
            final_result[final_result_channel_step+row] = scratchpad_memory[scratchpad_max_value_index_step + i];
    }


    channel_scratchpad_memory_step = result_size.y * channel;
    scratchpad_memory[channel_scratchpad_memory_step + row] = final_result[final_result_channel_step + row];

//    printf("CHECKPOINT MAX VALUE: channel: %d row: %d scratchpad_memory[channel_scratchpad_memory_step+row]: %f step: %d calculated: %f \n",
//           channel, row, scratchpad_memory[channel_scratchpad_memory_step+row], channel_scratchpad_memory_step + row, final_result[final_result_channel_step + row]);
}

__global__ void gpu_softmax_reduce_sum_elements_row(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    //Shared working memory
    extern __shared__ float sh_buffer[];

    float* sum_value = sh_buffer;
    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    const int channel_shared_memory_step = blockWidth * blockHeight;
    const int sum_value_index = channel_shared_memory_step * threadIdx.z + threadIdx.y * blockWidth + threadIdx.x;
    sum_value[sum_value_index] = 0;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c) return;

    const int scratchpad_channel_memory_step = a_info.h;
    const float max_value = scratchpad_memory[scratchpad_channel_memory_step * channel + row];

    float* ptr = a_input + channel * a_info.cstep + row * a_info.w + column;
    *ptr = static_cast<float>(exp(*ptr - max_value));

    sum_value[sum_value_index] = *ptr;

    __syncthreads();

    for (int i = (blockWidth+1) / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {
            sum_value[sum_value_index] += sum_value[sum_value_index + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        //TODO check possible overwrite of scratchpad in case of unbalanced block execution
        const int scratchpad_memory_offset_sum = a_info.h * a_info.c;
        const int channel_scratchpad_memory_step = gridDim.x * a_info.h;
        const int scracthpad_index = scratchpad_memory_offset_sum + channel * channel_scratchpad_memory_step + row * gridDim.x + blockIdx.x;
        scratchpad_memory[scracthpad_index] = sum_value[sum_value_index];

//        printf("CHECKPOINT SUM VALUE: channel: %d row: %d scratchpad_memory[scracthpad_index] : %f \n", channel, row, scratchpad_memory[scracthpad_index] );
    }
}

__global__ void gpu_softmax_reduce_sum_elements_row_result( float* scratchpad_memory, const dim3 result_size) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float sh_buffer[];
    float *final_result = sh_buffer;
    const int final_result_channel_offset = result_size.y * threadIdx.z;
    final_result[final_result_channel_offset + row] = 0;

    if (column >= result_size.x || row >= result_size.y || channel >= result_size.z) return;

    const int scratchpad_memory_offset_sum = result_size.y * result_size.z;
    const int channel_scratchpad_memory_step = result_size.x * result_size.y;
    const int scratchpad_max_value_index_step = scratchpad_memory_offset_sum + channel_scratchpad_memory_step * channel + row * result_size.x;

    for (int i = 0; i < result_size.x; i++)
    {
            final_result[final_result_channel_offset + row] = final_result[final_result_channel_offset + row] + scratchpad_memory[scratchpad_max_value_index_step + i];
    }

    //TODO check possible overwrite of scratchpad in case of unbalanced block execution
    const int scratchpad_memory_channel_result_step = result_size.y;
    scratchpad_memory[scratchpad_memory_channel_result_step * channel + row] = final_result[final_result_channel_offset + row];

//    printf("CHECKPOINT SUM VALUE: channel: %d row: %dscratchpad_memory[scratchpad_memory_channel_result_step * channel + row] : %f \n", channel, row,
//           scratchpad_memory[scratchpad_memory_channel_result_step * channel + row] );
}

__global__ void gpu_softmax_divide_elements_row(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory) {

    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c) return;

    const int channel_offset = a_info.h;
    const float sum = scratchpad_memory[channel_offset * channel + row];
    float* ptr = a_input + channel * a_info.cstep + row * a_info.w + column;
    *ptr = *ptr / sum;
}


__global__ void gpu_softmax_reduce_find_max_column(const float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    //Shared working memory
    extern __shared__ float sh_buffer[];

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int index = channel * a_info.cstep + row * a_info.w + column;

    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    const int channel_shared_memory_step = blockDim.x * (blockDim.y);

    float* max_value = sh_buffer;
    const int max_column_value_index = channel_shared_memory_step * threadIdx.z + threadIdx.y * blockWidth + threadIdx.x;
    max_value[max_column_value_index] = -NPP_MAXABS_32F;

    if (row >= a_info.h || column >= a_info.w || channel >= a_info.c) {
        return;
    }

    max_value[max_column_value_index] = a_input[index];
    __syncthreads();

    for (int j = (blockHeight + 1) / 2; j > 0; j /= 2)
    {
        if (threadIdx.y < j)
        {
            if (max_value[max_column_value_index] < max_value[max_column_value_index + j*blockWidth]) {
                max_value[max_column_value_index] = max_value[max_column_value_index + j*blockWidth];
            }
        }
        __syncthreads();
    }

    if (threadIdx.y == 0)
    {
        const int scratchpad_memory_offset_sum = a_info.w * a_info.c;
        const int scratchpad_channel_memory_step = gridDim.y * a_info.w;
        const int scracthpad_index = scratchpad_memory_offset_sum + channel * scratchpad_channel_memory_step + column * gridDim.y + blockIdx.y;
        scratchpad_memory[scracthpad_index] = max_value[max_column_value_index];
    }
}

__global__ void gpu_softmax_reduce_find_max_column_result( float* scratchpad_memory, dim3 result_size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float sh_buffer[];
    float *final_result = sh_buffer;
    const int final_result_channel_step = result_size.x * threadIdx.z;
    final_result[final_result_channel_step + column] = -NPP_MAXABS_32F;

    if (column >= result_size.x || row >= result_size.y || channel >= result_size.z) return;

    const int scratchpad_memory_offset_sum = result_size.x * result_size.z;
    const int scratchpad_channel_memory_step = result_size.x * result_size.y;
    const int scratchpad_max_value_index_step = scratchpad_memory_offset_sum + channel * scratchpad_channel_memory_step + column * result_size.y;

    for (int i = 0; i < result_size.y; i++)
    {
        if (final_result[final_result_channel_step +column] < scratchpad_memory[scratchpad_max_value_index_step + i])
            final_result[final_result_channel_step +column] = scratchpad_memory[scratchpad_max_value_index_step + i];
    }

    const int channel_scratchpad_memory_step = result_size.x * channel;
    scratchpad_memory[channel_scratchpad_memory_step + column] = final_result[final_result_channel_step + column];

//    printf("CHECKPOINT MAX VALUE: channel: %d column: %d scratchpad_memory[channel_scratchpad_memory_step + column] : %f \n",
//           channel, column, scratchpad_memory[channel_scratchpad_memory_step + column] );
}

__global__ void gpu_softmax_reduce_sum_elements_column(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    //Shared working memory
    extern __shared__ float sh_buffer[];

    float* sum_value = sh_buffer;
    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    const int channel_shared_memory_step = blockWidth * blockHeight;
    const int column_value_index = channel_shared_memory_step * threadIdx.z + threadIdx.y * blockWidth + threadIdx.x;
    sum_value[column_value_index] = 0;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c) {
        return;
    }

    const int scratchpad_channel_memory_step = a_info.w;
    const float max_value = scratchpad_memory[scratchpad_channel_memory_step * channel + column];
    float* ptr = a_input + channel * a_info.cstep + row * a_info.w + column;
    *ptr = static_cast<float>(exp(*ptr - max_value));

    sum_value[column_value_index] = *ptr;

    __syncthreads();

    for (int j = (blockHeight + 1) / 2; j > 0; j /= 2)
    {
        if (threadIdx.y < j)
        {
            sum_value[column_value_index] += sum_value[column_value_index + j * blockWidth];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0)
    {
        const int scratchpad_memory_offset_sum = a_info.w * a_info.c;
        const int channel_scratchpad_memory_step = gridDim.y * a_info.w;
        const int scracthpad_index = scratchpad_memory_offset_sum + channel * channel_scratchpad_memory_step + column * gridDim.y + blockIdx.y;
        scratchpad_memory[scracthpad_index] = sum_value[column_value_index];

//        printf("CHECKPOINT SUM VALUE: channel: %d column: %d row: %d scratchpad_memory[scracthpad_index]  : %f \n",
//                   channel, column, row, scratchpad_memory[scracthpad_index]  );
    }
}

__global__ void gpu_softmax_reduce_sum_elements_column_result( float* scratchpad_memory, const dim3 result_size) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float sh_buffer[];
    float *final_result = sh_buffer;
    const int final_result_channel_offset = result_size.x * threadIdx.z;
    final_result[final_result_channel_offset + column] = 0;

    if (column >= result_size.x || row >= result_size.y || channel >= result_size.z) return;

    const int scratchpad_memory_offset_sum = result_size.x * result_size.z;
    const int channel_scratchpad_memory_step = result_size.x * result_size.y;
    const int scratchpad_max_value_index_step = scratchpad_memory_offset_sum + channel_scratchpad_memory_step * channel + column * result_size.y;

    for (int i = 0; i < result_size.y; i++)
    {
        final_result[final_result_channel_offset + column] = final_result[final_result_channel_offset + column] + scratchpad_memory[scratchpad_max_value_index_step + i];
    }

    const int scratchpad_memory_channel_result_step = result_size.x;
    scratchpad_memory[scratchpad_memory_channel_result_step * channel + column] = final_result[final_result_channel_offset + column];

//    if (channel == 0)
//    printf("CHECKPOINT SUM VALUE: channel: %d column: %d scratchpad_memory[scratchpad_memory_channel_result_step * channel + column]  : %f \n",
//           channel, column, scratchpad_memory[scratchpad_memory_channel_result_step * channel + column]  );
}

__global__ void gpu_softmax_divide_elements_column(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory) {

    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c) return;

    const int channel_offset = a_info.w;
    const float sum = scratchpad_memory[channel_offset * channel + column];
    float* ptr = a_input + channel * a_info.cstep + row * a_info.w + column;
    *ptr = *ptr / sum;
}



__global__ void gpu_softmax_reduce_find_max_channel(const float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    //Shared working memory
    extern __shared__ float sh_buffer[];

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int max_index = row * a_info.w + column;

    if (row >= a_info.h || column >= a_info.w || channel >= a_info.c) {
        return;
    }

    scratchpad_memory[max_index] = -NPP_MAXABS_32F;

    for (int c = 0; c < a_info.c; ++c)
    {
        const float* current_ptr = a_input + c * a_info.cstep + row * a_info.w + column;
        if (scratchpad_memory[max_index]  < *current_ptr)
            scratchpad_memory[max_index] = *current_ptr;
    }

//    if (column == 0 && row==0)
//        printf("CHECKPOINT MAX VALUE: channel: %d scratchpad_memory[index]: %f index: %d \n",
//               channel, scratchpad_memory[index], index);

}



__global__ void gpu_softmax_reduce_calculates_elements_channel(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c) {
        return;
    }

    const int index = row * a_info.w + column;

    const float max_value = scratchpad_memory[index];

    float* ptr = a_input + channel * a_info.cstep + row * a_info.w + column;
    *ptr = static_cast<float>(exp(*ptr - max_value));

//    if (row == 0 && column == 0)
//        printf("CHECKPOINT ZERO CALCULATED VALUE: channel: %d scratchpad_memory[index]: %f max: %f index: %d\n",
//               channel, *ptr, max_value, index);

}

__global__ void gpu_softmax_reduce_sum_elements_channel(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    const int index = row * a_info.w + column;
    scratchpad_memory[index] = 0;

    for (int c = 0; c < a_info.c; ++c)
    {
        float* ptr = a_input + c * a_info.cstep + row * a_info.w + column;
        scratchpad_memory[index] += *ptr;
    }

//    if (channel == 0)
//        printf("CHECKPOINT SUM VALUE: row: %d column: %d scratchpad_memory[index]  : %f \n",
//               row, column, scratchpad_memory[index]);

}

__global__ void gpu_softmax_divide_elements_channel(float* a_input, const ncnn::CudaMatInfo a_info, float* scratchpad_memory) {

    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c) return;

    const int sum_index = row * a_info.w + column;
    const int index = channel * a_info.cstep + row * a_info.w + column;

    const float sum = scratchpad_memory[sum_index];
    float* ptr = a_input + index;
    *ptr = *ptr / sum;
}

namespace ncnn {

int softmax_cuda_forward_inplace(float* a_input, const ncnn::CudaMatInfo& a_info,
                                 const int axis,
                                 float* gpu_scratchpad_memory,
                                 int gpu_scratchpad_memory_size)
{
    if ((a_info.dims == 1) || (a_info.dims == 2 && axis == 1) || (a_info.dims == 3 && axis == 2)) {
        checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));
        int thread_per_block_x = ((a_info.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = a_info.h;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = a_info.c;
        const int total_number_of_columns = a_info.w;
        const int total_number_of_rows = a_info.h;
        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_softmax_reduce_find_max_row<<<grid_size, block_size, (sizeof(float) * thread_per_block_x * thread_per_block_y * thread_per_block_z)>>>
            (a_input, a_info, gpu_scratchpad_memory);

        const dim3 block_size_reduce(1, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size_reduce(1, (total_number_of_rows - 1) / thread_per_block_y + 1,
                                    (total_number_of_channels - 1) / thread_per_block_z + 1);
        const dim3 result_size(grid_size.x, a_info.h, a_info.c);
        gpu_softmax_reduce_find_max_row_result<<<grid_size_reduce, block_size_reduce, sizeof(float) * total_number_of_rows * thread_per_block_z * 2>>>
            (gpu_scratchpad_memory, result_size);

        gpu_softmax_reduce_sum_elements_row<<<grid_size, block_size, (sizeof(float) * thread_per_block_x * thread_per_block_y * thread_per_block_z)>>>
            (a_input, a_info, gpu_scratchpad_memory);

        gpu_softmax_reduce_sum_elements_row_result<<<grid_size_reduce, block_size_reduce, sizeof(float) * total_number_of_rows * thread_per_block_z * 2>>>
            (gpu_scratchpad_memory, result_size);

        gpu_softmax_divide_elements_row<<<grid_size, block_size>>>(a_input, a_info, gpu_scratchpad_memory);

    } else if ((a_info.dims == 2 && axis == 0) || (a_info.dims == 3 && axis == 1)) {
        checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));
        int thread_per_block_x = ((a_info.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((a_info.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = a_info.c;
        const int total_number_of_columns = a_info.w;
        const int total_number_of_rows = a_info.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_softmax_reduce_find_max_column<<<grid_size, block_size, (sizeof(float) * thread_per_block_x * thread_per_block_y * thread_per_block_z)>>>
            (a_input, a_info, gpu_scratchpad_memory);

        const dim3 block_size_reduce(thread_per_block_x, 1, thread_per_block_z);
        const dim3 grid_size_reduce((total_number_of_columns - 1) / thread_per_block_x + 1, 1,
                                    (total_number_of_channels - 1) / thread_per_block_z + 1);
        const dim3 result_size(a_info.w, grid_size.y, a_info.c);
        gpu_softmax_reduce_find_max_column_result<<<grid_size_reduce, block_size_reduce,
                                                    sizeof(float) * total_number_of_columns*thread_per_block_z * 2>>>(gpu_scratchpad_memory, result_size);

        gpu_softmax_reduce_sum_elements_column<<<grid_size, block_size, (sizeof(float) * thread_per_block_x * thread_per_block_y * thread_per_block_z)>>>
            (a_input, a_info, gpu_scratchpad_memory);

        gpu_softmax_reduce_sum_elements_column_result<<<grid_size_reduce, block_size_reduce, sizeof(float) * total_number_of_columns*thread_per_block_z * 2>>>
            (gpu_scratchpad_memory, result_size);

        gpu_softmax_divide_elements_column<<<grid_size, block_size>>>(a_input, a_info, gpu_scratchpad_memory);

    }  else if (a_info.dims == 3 && axis == 0) {
        checkCudaErrors(cudaMemset(gpu_scratchpad_memory, 0, gpu_scratchpad_memory_size));
        int thread_per_block_x = ((a_info.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((a_info.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = a_info.c;
        const int total_number_of_columns = a_info.w;
        const int total_number_of_rows = a_info.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             1);
        gpu_softmax_reduce_find_max_channel<<<grid_size, block_size>>>(a_input, a_info, gpu_scratchpad_memory);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        const dim3 block_size_calculate(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size_calculate((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1, total_number_of_channels);
        gpu_softmax_reduce_calculates_elements_channel<<<grid_size_calculate, block_size_calculate>>>(a_input, a_info, gpu_scratchpad_memory);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        const dim3 block_size_sum(thread_per_block_x, thread_per_block_y, 1);
        const dim3 grid_size_sum((total_number_of_columns - 1) / thread_per_block_x + 1,
                                       (total_number_of_rows - 1) / thread_per_block_y + 1, 1);
        gpu_softmax_reduce_sum_elements_channel<<<grid_size_sum, block_size_sum>>>(a_input, a_info, gpu_scratchpad_memory);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        const dim3 block_size_div(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size_div((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                                 (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_softmax_divide_elements_channel<<<grid_size_div, block_size_div>>>(a_input, a_info, gpu_scratchpad_memory);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }


    return 0;
}



}
