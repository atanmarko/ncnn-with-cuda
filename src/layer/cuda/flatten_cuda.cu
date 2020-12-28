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

#include "mat.h"


__global__ void gpu_flatten_forward(const unsigned char* a_input, const ncnn::CudaMatInfo a_info, unsigned char* output) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    const int input_index = channel * a_info.cstep * a_info.elemsize + row * a_info.w * a_info.elemsize + column * a_info.elemsize;
    const int output_index = a_info.w * a_info.h * a_info.elemsize * channel + row * a_info.w * a_info.elemsize + column * a_info.elemsize;
    memcpy((void*)(output + output_index), (void*)(a_input + input_index), a_info.elemsize);

}



namespace ncnn {

int flatten_cuda_forward(const unsigned char* bottom_blob, const ncnn::CudaMatInfo bottom_blob_info,
                         unsigned char* top_blob)
{
    int thread_per_block_x = ((bottom_blob_info.w - 1) / 64 + 1) * 64;
    if (thread_per_block_x > 128) thread_per_block_x = 128;
    int thread_per_block_y = ((bottom_blob_info.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    const int thread_per_block_z = 1;
    const int total_number_of_channels = bottom_blob_info.c;
    const int total_number_of_columns = bottom_blob_info.w;
    const int total_number_of_rows = bottom_blob_info.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    gpu_flatten_forward<<<grid_size, block_size>>>(bottom_blob, bottom_blob_info, top_blob);

    return 0;
}


}