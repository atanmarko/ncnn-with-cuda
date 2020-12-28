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


__global__ void gpu_reshape_type1(const unsigned char* a_input, const ncnn::CudaMatInfo a_info, unsigned char* output,
                                                   const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    const int input_channel_step = a_info.cstep * channel * a_info.elemsize;
    const int output_channel_step = a_info.w * a_info.h * a_info.elemsize * channel;

    const unsigned char* input_ptr = a_input + input_channel_step + row * a_info.w * a_info.elemsize + column * a_info.elemsize;
    unsigned char* output_ptr = output + output_channel_step + row * a_info.w * a_info.elemsize + column * a_info.elemsize;

    memcpy(output_ptr, input_ptr, a_info.elemsize);
}

__global__ void gpu_reshape_type2(const unsigned char* a_input, const ncnn::CudaMatInfo a_info, unsigned char* output,
                                  const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= output_info.w || row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    const int input_channel_step =  output_info.w * output_info.h * output_info.elemsize * channel;
    const int output_channel_step = output_info.cstep * output_info.elemsize * channel;

    const unsigned char* input_ptr = a_input + input_channel_step + row * output_info.w * output_info.elemsize + column * output_info.elemsize;
    unsigned char* output_ptr = output + output_channel_step + row * output_info.w * output_info.elemsize + column * output_info.elemsize;

    memcpy(output_ptr, input_ptr, output_info.elemsize);
}



__global__ void gpu_reshape_forward_need_permute_2(const float* a_input, const ncnn::CudaMatInfo a_info, float* output,
                                                 const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column >= a_info.w || row >= a_info.h)
    {
        return;
    }

    output[column * a_info.h + row] = a_input[row * a_info.w + column];
}

__global__ void gpu_reshape_forward_need_permute_3(const float* a_input, const ncnn::CudaMatInfo a_info, float* output,
                                                   const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= a_info.w || row >= a_info.h || channel >= a_info.c)
    {
        return;
    }

    int input_channel_step = channel * a_info.cstep;

    int output_column = channel;
    int output_row = column;
    int output_channel = row;
    int output_channel_step = output_channel * output_info.cstep;

    output[output_channel_step + output_row * output_info.w + output_column] = a_input[input_channel_step + row * a_info.w + column];
}


__global__ void gpu_reshape_forward_need_permute_3_v2(const float* a_input, const ncnn::CudaMatInfo a_info, float* output,
                                                   const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= output_info.w || row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    int input_column = channel;
    int input_row = column;
    int input_channel = row;
    int input_channel_step = input_channel * a_info.cstep;
    int output_channel_step = channel * output_info.cstep;

    output[output_channel_step + row * output_info.w + column] = a_input[input_channel_step + input_row * a_info.w + input_column];
}


namespace ncnn {

int reshape_cuda_forward(const ncnn::CudaMat& bottom_blob, ncnn::CudaMat& top_blob, const int outw, const int outh, const int outc, bool need_permute, int ndim)
{
    const int dims = bottom_blob.dims;
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    const CudaMatInfo bottom_blob_info{bottom_blob};

    if (need_permute) {
        CudaMat bottom_blob_permuted = bottom_blob;

        if (dims == 2)
        {
            bottom_blob_permuted.create(bottom_blob.w, bottom_blob.h, bottom_blob.elemsize, cuda_allocator);
            if (bottom_blob_permuted.empty())
                return -100;

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

            const CudaMatInfo bottom_blob_permuted_info{bottom_blob_permuted};

            gpu_reshape_forward_need_permute_2<<<grid_size, block_size>>>(static_cast<const float *>(bottom_blob.get_craw_data()),
                                                                            bottom_blob_info,
                                                                            static_cast<float*>(bottom_blob_permuted.get_raw_data()),
                                                                          bottom_blob_permuted_info);
        }

        if (dims == 3)
        {
            bottom_blob_permuted.create(bottom_blob.c, bottom_blob.w, bottom_blob.h, bottom_blob.elemsize, cuda_allocator);
            if (bottom_blob_permuted.empty())
                return -100;

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

            const CudaMatInfo bottom_blob_permuted_info{bottom_blob_permuted};

            gpu_reshape_forward_need_permute_3<<<grid_size, block_size>>>(static_cast<const float *>(bottom_blob.get_craw_data()),
                                                                          bottom_blob_info,
                                                                          static_cast<float*>(bottom_blob_permuted.get_raw_data()),
                                                                          bottom_blob_permuted_info);
        }

        if (ndim == 1)
        {
            top_blob = bottom_blob_permuted.reshape(outw, cuda_allocator);
            if (top_blob.empty())
                return -100;

            return 0;
        }

        // permute on nhwc/nhc
        CudaMat top_blob_permuted;
        if (ndim == 2)
        {
            top_blob_permuted = bottom_blob_permuted.reshape(outh, outw, cuda_allocator);
        }
        if (ndim == 3)
        {
            top_blob_permuted = bottom_blob_permuted.reshape(outc, outw, outh, cuda_allocator);
        }

        if (top_blob_permuted.empty())
            return -100;

        if (ndim == 2)
        {
            // wh -> hw
            top_blob.create(outw, outh, bottom_blob.elemsize, cuda_allocator);
            if (top_blob.empty())
                return -100;

            const CudaMatInfo top_blob_permuted_info{top_blob_permuted};
            const CudaMatInfo top_blob_info{bottom_blob};

            int thread_per_block_x = ((top_blob_permuted.w - 1) / 64 + 1) * 64;
            if (thread_per_block_x > 128) thread_per_block_x = 128;
            int thread_per_block_y = ((top_blob_permuted.h - 1) / 8 + 1) * 8;
            if (thread_per_block_y > 8) thread_per_block_y = 8;
            const int thread_per_block_z = 1;
            const int total_number_of_channels = top_blob_permuted.c;
            const int total_number_of_columns = top_blob_permuted.w;
            const int total_number_of_rows = top_blob_permuted.h;

            const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
            const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                                 (total_number_of_rows - 1) / thread_per_block_y + 1,
                                 (total_number_of_channels - 1) / thread_per_block_z + 1);

            gpu_reshape_forward_need_permute_2<<<grid_size, block_size>>>(static_cast<const float *>(top_blob_permuted.get_craw_data()),
                                                                          top_blob_permuted_info,
                                                                          static_cast<float*>(top_blob.get_raw_data()),
                                                                          top_blob_info);
        }

        if (ndim == 3)
        {
            // chw -> hwc
            top_blob.create(outw, outh, outc, bottom_blob.elemsize, cuda_allocator);
            if (top_blob.empty())
                return -100;

            const CudaMatInfo top_blob_permuted_info{top_blob_permuted};
            const CudaMatInfo top_blob_info{top_blob};

            int thread_per_block_x = ((top_blob.w - 1) / 64 + 1) * 64;
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


            gpu_reshape_forward_need_permute_3_v2<<<grid_size, block_size>>>(static_cast<const float *>(top_blob_permuted.get_craw_data()),
                                                                          top_blob_permuted_info,
                                                                          static_cast<float*>(top_blob.get_raw_data()),
                                                                          top_blob_info);
        }

        return 0;
    } else {

        if (ndim == 1)
        {
            top_blob = bottom_blob.reshape(outw, cuda_allocator);
        }
        if (ndim == 2)
        {
            top_blob = bottom_blob.reshape(outw, outh, cuda_allocator);
        }
        if (ndim == 3)
        {
            top_blob = bottom_blob.reshape(outw, outh, outc, cuda_allocator);
        }

        if (top_blob.empty())
            return -100;

    }

    return 0;
}

int reshape_cuda_mat(const ncnn::CudaMat& input, ncnn::CudaMat& output, int type) {

    const CudaMatInfo input_info{input};
    const CudaMatInfo output_info{output};

    if (type == 1)
    {
        int thread_per_block_x = ((input.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((input.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        if (input.h == 1) thread_per_block_y = 1;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = input.c;
        const int total_number_of_columns = input.w;
        const int total_number_of_rows = input.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_reshape_type1<<<grid_size, block_size>>>(static_cast<const unsigned char*>(input.get_craw_data()),
                                                       input_info,
                                                       static_cast<unsigned char*>(output.get_raw_data()),
                                                       output_info);
    } else if (type == 2) {
        int thread_per_block_x = ((output_info.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = ((output_info.h - 1) / 8 + 1) * 8;
        if (thread_per_block_y > 8) thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = output_info.c;
        const int total_number_of_columns = output_info.w;
        const int total_number_of_rows = output_info.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);
        gpu_reshape_type2<<<grid_size, block_size>>>(static_cast<const unsigned char*>(input.get_craw_data()),
            input_info,
            static_cast<unsigned char*>(output.get_raw_data()),
            output_info);
    }

    return 0;
}


}