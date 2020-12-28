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


__device__ static  inline signed char gpu_float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

__global__ void gpu_quantize_forward(const float* a_input, const ncnn::CudaMatInfo a_info, signed char* output,
                                     const ncnn::CudaMatInfo output_info, const float scale) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    const int input_index = channel * a_info.cstep + row * a_info.w + column;
    const int output_index = channel * output_info.cstep + row * output_info.w + column;
    output[output_index] = gpu_float2int8(a_input[input_index] * scale);

}



namespace ncnn {

int quantize_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, float scale, std::shared_ptr<ncnn::CudaAllocator> blob_cuda_allocator = nullptr)
{
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator;

    if (blob_cuda_allocator.use_count() > 0)
        cuda_allocator = blob_cuda_allocator;
    else
        cuda_allocator = ncnn::get_current_gpu_allocator();

    if (bottom_blob.dims == 1)
    {
        top_blob.create(bottom_blob.w, (size_t)1u, cuda_allocator);
        if (top_blob.empty())
            return -100;
    }
    else if (bottom_blob.dims == 2)
    {
        top_blob.create(bottom_blob.w, bottom_blob.h, (size_t)1u, cuda_allocator);
        if (top_blob.empty())
            return -100;
    }
    else if (bottom_blob.dims == 3)
    {
        top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, (size_t)1u, cuda_allocator);
        if (top_blob.empty())
            return -100;
    }

    int thread_per_block_x = ((bottom_blob.w - 1) / 64 + 1) * 64;
    if (thread_per_block_x > 128) thread_per_block_x = 128;
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

    const CudaMatInfo bottom_blob_info{bottom_blob};
    const CudaMatInfo top_blob_info{top_blob};

    gpu_quantize_forward<<<grid_size, block_size>>>(static_cast<const float *>(bottom_blob.get_craw_data()),
                                                      bottom_blob_info,
                                                      static_cast<signed char *>(top_blob.get_raw_data()),
                                                      top_blob_info,
                                                      scale);

    return 0;
}


}