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
#include "packing_cuda.h"

#include <iostream>


__global__ void gpu_packing_forward_2(const float* a_input, const ncnn::CudaMatInfo input_info,
                                      float* d_output, const ncnn::CudaMatInfo output_info,
                                      const ncnn::Packing_cuda::packing_options options)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= output_info.h || column >= input_info.w) return;

    float* output_channel_ptr = (float*)((unsigned char*)d_output); //out
    unsigned char* outptr = (unsigned char*)output_channel_ptr + row * input_info.w * output_info.elemsize;
    unsigned char* out_elem_ptr = outptr + column * output_info.elemsize;
    size_t lane_size = output_info.elemsize / output_info.elempack;

    for (int k = 0; k < output_info.elempack; k++)
    {
        int srcy = (row * output_info.elempack + k) / input_info.elempack;
        if (srcy >= input_info.h)
            break;

        const int srck = (row * output_info.elempack + k) % input_info.elempack;
        const unsigned char* input_ptr = (const unsigned char*)a_input + srcy * input_info.w * input_info.elemsize;
        const unsigned char* elem_ptr = input_ptr + column * input_info.elemsize;

        memcpy(out_elem_ptr+ k * lane_size, elem_ptr + srck * lane_size, lane_size);
    }
}

__global__ void gpu_packing_forward_3(const float* a_input, const ncnn::CudaMatInfo input_info,
                                      float* d_output, const ncnn::CudaMatInfo output_info,
                                      const ncnn::Packing_cuda::packing_options options)
{
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column >= input_info.w || row >= input_info.h || channel >= output_info.c) return;

    const int output_channel_step = channel * output_info.cstep * output_info.elemsize;
    float* output_channel_ptr = (float*)((unsigned char*)d_output + output_channel_step); //out
    unsigned char* outptr = (unsigned char*)output_channel_ptr + row * output_info.w * output_info.elemsize;
    unsigned char* out_elem_ptr = outptr + column * output_info.elemsize;
    size_t lane_size = output_info.elemsize / output_info.elempack;

    for (int k = 0; k < output_info.elempack; k++)
    {
        int srcq = (channel * output_info.elempack + k) / input_info.elempack;
        if (srcq >= input_info.c)
            break;

        const int input_channel_step = srcq * input_info.cstep * input_info.elemsize;
        float* input_channel_ptr = (float*)((unsigned char*)a_input + input_channel_step); //m
        const unsigned char* input_ptr = (const unsigned char*)input_channel_ptr + row * input_info.w * input_info.elemsize;
        const unsigned char* elem_ptr = input_ptr + column * input_info.elemsize;

        const int srck = (channel * output_info.elempack + k) % input_info.elempack;
        memcpy(out_elem_ptr+ k * lane_size, elem_ptr + srck * lane_size, lane_size);
    }

}

namespace ncnn {

int packing_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Packing_cuda::packing_options options)
{
    int elempack = bottom_blob.elempack;

    if (elempack == options.out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;
    const size_t elemsize = bottom_blob.elemsize;

    if (!options.use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % options.out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % options.out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * elempack % options.out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    if (dims == 1)
    {
        if (options.out_elempack == 1)
        {
            top_blob = bottom_blob;
            top_blob.w = w * elempack;
            top_blob.cstep = w * elempack;
            top_blob.elemsize = elemsize / elempack;
            top_blob.elempack = options.out_elempack;
            return 0;
        }

        int outw = (w * elempack + options.out_elempack - 1) / options.out_elempack;
        size_t out_elemsize = elemsize / elempack * options.out_elempack;


        top_blob.create(outw, out_elemsize, options.out_elempack, cuda_allocator);
        if (top_blob.empty())
            return -100;

        checkCudaErrors(cudaMemcpy(top_blob.data,bottom_blob.data, w * elemsize, cudaMemcpyDeviceToDevice));

        return 0;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + options.out_elempack - 1) / options.out_elempack;
        size_t out_elemsize = elemsize / elempack * options.out_elempack;

        top_blob.create(w, outh, out_elemsize, options.out_elempack, cuda_allocator);
        if (top_blob.empty())
            return -100;


        const ncnn::CudaMatInfo input_info{bottom_blob};
        const ncnn::CudaMatInfo output_info{top_blob};

        const int thread_per_block_x = 128;
        const int thread_per_block_y = 8;
        const int total_number_of_columns = input_info.w;
        const int total_number_of_rows = outh;
        const dim3 block_size(thread_per_block_x, thread_per_block_y, 1);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1, 1);

        gpu_packing_forward_2<<<grid_size, block_size>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                         input_info,
                                                         static_cast<float*>(top_blob.get_raw_data()),
                                                         output_info,
                                                         options);


        return 0;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + options.out_elempack - 1) / options.out_elempack;
        size_t out_elemsize = elemsize / elempack * options.out_elempack;
        size_t lane_size = out_elemsize / options.out_elempack;

        top_blob.create(w, h, outc, out_elemsize, options.out_elempack, cuda_allocator);
        if (top_blob.empty())
            return -100;

        const ncnn::CudaMatInfo input_info{bottom_blob};
        const ncnn::CudaMatInfo output_info{top_blob};

        const int thread_per_block_x = 64;
        const int thread_per_block_y = 8;
        const int thread_per_block_z = 2;
        const int total_number_of_channels = outc;
        const int total_number_of_columns = input_info.w;
        const int total_number_of_rows = input_info.h;
        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        gpu_packing_forward_3<<<grid_size, block_size>>>(static_cast<const float*>(bottom_blob.get_craw_data()),
                                                           input_info,
                                                           static_cast<float*>(top_blob.get_raw_data()),
                                                           output_info,
                                                           options);
    }

    return 0;
}



}
