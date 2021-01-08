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

#include "crop_cuda.h"


__global__ void gpu_crop_forward(const unsigned char* a_input, const ncnn::CudaMatInfo a_info, unsigned char* output, const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

}

template<typename T>
__global__ void gpu_crop_copy_cut_border_image(const T* a_input, const ncnn::CudaMatInfo a_info,
                                               T* output, const ncnn::CudaMatInfo output_info,
                                               const int top, const int left, const int channel_offset = 0) {

    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z  + threadIdx.z ;

    if ((row >= output_info.h)
        || (column >= output_info.w)
        || (channel >= output_info.c))
    {
        return;
    }

    const int input_row = row + top;
    const int input_column = column + left;
    const int input_channel = channel + channel_offset;

    const int input_channel_step = input_channel * a_info.cstep * a_info.elemsize;
    const int output_channel_step = channel * output_info.cstep * output_info.elemsize;

    const T* ptr = (T*)((unsigned char*)a_input + input_channel_step + a_info.w * input_row * a_info.elemsize + input_column * a_info.elemsize);
    T* out_ptr = (T*)((unsigned char*)output + output_channel_step + output_info.w * row * output_info.elemsize + column * output_info.elemsize);
    *out_ptr = *ptr;
}


namespace ncnn {

int crop_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Crop_cuda::Crop_info crop_info, const Option& opt)
{
    int input_w = bottom_blob.w;
    int input_h = bottom_blob.h;
    int input_channels = bottom_blob.c;
    int input_dims = bottom_blob.dims;
    size_t input_elemsize = bottom_blob.elemsize;

    const ncnn::CudaMatInfo bottom_blob_info{bottom_blob};
    ncnn::CudaMatInfo top_blob_info{top_blob};

    if (input_dims == 1) {
        if (crop_info.outw == input_w) {
            top_blob = bottom_blob;
        }
        top_blob.create(crop_info.outw, input_elemsize, opt.blob_cuda_allocator);
        if (top_blob.empty())
            return -100;

        top_blob_info = ncnn::CudaMatInfo{top_blob};
    }
    else if (input_dims == 2)
    {
        if (crop_info.outw == input_w && crop_info.outh == input_h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(crop_info.outw, crop_info.outh, input_elemsize, opt.blob_cuda_allocator);
        if (top_blob.empty())
            return -100;

        top_blob_info = ncnn::CudaMatInfo{top_blob};
    }
    else if (input_dims == 3)
    {
        if (crop_info.outw == input_w && crop_info.outh == input_h && crop_info.outc == input_channels)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(crop_info.outw, crop_info.outh, crop_info.outc, input_elemsize, opt.blob_cuda_allocator);
        if (top_blob.empty())
            return -100;

        top_blob_info = ncnn::CudaMatInfo{top_blob};
    }

    int thread_per_block_x = ((top_blob_info.w - 1) / 64 + 1) * 64;
    if (thread_per_block_x > 128) thread_per_block_x = 128;
    int thread_per_block_y = ((top_blob_info.h - 1) / 8 + 1) * 8;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    const int thread_per_block_z = 1;
    const int total_number_of_channels = top_blob_info.c;
    const int total_number_of_columns = top_blob_info.w;
    const int total_number_of_rows = top_blob_info.h;

    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);


    if (input_elemsize == 1)
        gpu_crop_copy_cut_border_image<signed char><<<grid_size, block_size>>>(static_cast<const signed char*>(bottom_blob.get_craw_data()),
                                                                               bottom_blob_info,
                                                                               static_cast<signed char*>(top_blob.get_raw_data()),
                                                                               top_blob_info,
                                                                               crop_info.hoffset, crop_info.woffset,
                                                                               (crop_info.coffset < 0 ? 0 : crop_info.coffset));
    if (input_elemsize == 2)
        gpu_crop_copy_cut_border_image<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(bottom_blob.get_craw_data()),
                                                                                  bottom_blob_info,
                                                                                  static_cast<unsigned short*>(top_blob.get_raw_data()),
                                                                                  top_blob_info,
                                                                                  crop_info.hoffset, crop_info.woffset,
                                                                                  (crop_info.coffset < 0 ? 0 : crop_info.coffset));
    if (input_elemsize == 4)
        gpu_crop_copy_cut_border_image<float><<<grid_size, block_size>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                                           bottom_blob_info,
                                                                           static_cast<float*>(top_blob.get_raw_data()),
                                                                           top_blob_info,
                                                                           crop_info.hoffset, crop_info.woffset,
                                                                         (crop_info.coffset < 0 ? 0 : crop_info.coffset));

    return 0;
}


}