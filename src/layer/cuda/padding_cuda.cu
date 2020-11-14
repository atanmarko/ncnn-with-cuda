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
#include "padding_cuda.h"

#include <iostream>



namespace ncnn {

template<typename T>
__global__ void gpu_copy_make_border_image_type0(const T* src, const CudaMatInfo src_info, T* dst, const CudaMatInfo dst_info,
                                             int top, int left, int type, T v)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
//    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (row >= dst_info.h || column >= dst_info.w)
        return;

    T* outptr = dst + row * dst_info.w + column;
    T output_value = v; // initialize whole matrix with padding

    if ((row >= top && row < top + src_info.h)
        && (column >= left && column < left + src_info.w))
    {
        const T* inptr = src + (row - top) * src_info.w + column - left;
        output_value = *inptr;
    }
    *outptr = output_value;
}

template<typename T>
__global__ void gpu_copy_make_border_image_type1(const T* src, const CudaMatInfo src_info, T* dst, const CudaMatInfo dst_info,
                                             int top, int left, int type, T v)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dst_info.h || column >= dst_info.w)
        return;

    T* outptr = dst + row * dst_info.w + column;
    T output_value = 0;

    if (row < top) {
        if (column < left )
            output_value = src[0];
        else if (column >= left + src_info.w)
            output_value = src[src_info.w - 1];
        else
            output_value = src[column - left];
    }
    else if (row >= top && row < top + src_info.h)
    {
        const T* inptr = src + (row - top) * src_info.w;
        if (column < left)
            output_value = inptr[0];
        else if (column >= left+src_info.w)
            output_value = inptr[src_info.w - 1];
        else
            output_value = inptr[column - left];
    }
    else if (row >= top + src_info.h) {
        const T* inptr = src + (src_info.h - 1) * src_info.w;
        if (column < left)
            output_value = *inptr;
        else if (column >= left + src_info.w)
            output_value = inptr[src_info.w - 1];
        else
            output_value = inptr[column - left];
    }

    *outptr = output_value;
}


template<typename T>
__global__ void gpu_copy_make_border_image_type2(const T* src, const CudaMatInfo src_info, T* dst, const CudaMatInfo dst_info,
                                             int top, int left, int type, T v)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= dst_info.h || column >= dst_info.w)
        return;

    T* outptr = dst + row * dst_info.w + column;
    T output_value = 0;

    if (row < top) {
        const T* inptr = src + (top - row) * src_info.w;
        if (column < left )
            output_value = inptr[left - column];
        else if (column >= left && column < left + src_info.w)
            output_value = inptr[column - left];
        else if (column < dst_info.w)
            output_value = inptr[src_info.w - (column - left - src_info.w) - 2];
    }
    else if (row >= top && row < top + src_info.h)
    {
        const T* inptr = src + (row - top) * src_info.w;
        if (column < left)
            output_value = inptr[left - column];
        else if (column >= left+src_info.w)
            output_value = inptr[src_info.w - (column - left - src_info.w) - 2];
        else
            output_value = inptr[column - left];
    }
    else if (row >= top + src_info.h) {
        int diff = dst_info.h - top - src_info.h;
        const T* inptr = src + (src_info.h - (diff - (dst_info.h-row)) - 2) * src_info.w;
        if (column < left)
            output_value = inptr[left - column];
        else if (column >= left + src_info.w)
            output_value = inptr[src_info.w - (column - left - src_info.w) - 2];
        else
            output_value = inptr[column - left];
    }

    *outptr = output_value;
}


template<typename T>
__global__ void gpu_copy_make_border_image_3d_type0(const T* src, const CudaMatInfo src_info, T* dst, const CudaMatInfo dst_info,
                                             int front, int top, int left, int type, ncnn::GPUPaddingValue<T> values)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (row >= dst_info.h || column >= dst_info.w || channel >= dst_info.c)
        return;

    const int dst_channel_step = dst_info.cstep * channel;
    T* outptr = dst + dst_channel_step + row * dst_info.w + column;

    T output_value{};
    T padding_value{};
    if (values.per_channel_pad_data_size)
        padding_value = values.per_channel_values[channel];
    else
        padding_value = values.value;


    output_value = padding_value;

    if (channel < front || channel >= src_info.c+front) {
        //do nothing
    }
    else if ((row >= top && row < top + src_info.h)
        && (column >= left && column < left + src_info.w))
    {
        const T* inptr = src + (channel - front) * src_info.cstep + (row - top) * src_info.w + column - left;
        output_value = *inptr;
    }
    *outptr = output_value;
}

template<typename T>
__global__ void gpu_copy_make_border_image_3d_type1(const T* src, const CudaMatInfo src_info, T* dst, const CudaMatInfo dst_info,
                                                int front, int top, int left, int type, ncnn::GPUPaddingValue<T> values)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;


    if (row >= dst_info.h || column >= dst_info.w || channel >= dst_info.c)
        return;

    int q = channel - front;
    q = q <= 0 ? 0 : q;
    q = q >= src_info.c - 1 ? src_info.c - 1 : q;

    const int dst_channel_step = dst_info.cstep * channel;
    const int src_channel_step = src_info.cstep * q;

    T* outptr = dst + dst_channel_step + row * dst_info.w + column;


    T padding_value{};
    if (values.per_channel_pad_data_size)
        padding_value = values.per_channel_values[channel];
    else
        padding_value = values.value;

    T output_value = padding_value;

    if (row < top) {
        const T* inptr = src + src_channel_step;
        if (column < left )
            output_value = inptr[0];
        else if (column >= left + src_info.w)
            output_value = inptr[src_info.w - 1];
        else
            output_value = inptr[column - left];
    }
    else if (row >= top && row < top + src_info.h)
    {
        const T* inptr = src + src_channel_step + (row - top) * src_info.w;
        if (column < left)
            output_value = inptr[0];
        else if (column >= left+src_info.w)
            output_value = inptr[src_info.w - 1];
        else
            output_value = inptr[column - left];
    }
    else if (row >= top + src_info.h) {
        const T* inptr = src + src_channel_step +  (src_info.h - 1) * src_info.w;
        if (column < left)
            output_value = *inptr;
        else if (column >= left + src_info.w)
            output_value = inptr[src_info.w - 1];
        else
            output_value = inptr[column - left];
    }

    *outptr = output_value;
}


template<typename T>
__global__ void gpu_copy_make_border_image_3d_type2(const T* src, const CudaMatInfo src_info, T* dst, const CudaMatInfo dst_info,
                                                int front, int top, int left, int type, ncnn::GPUPaddingValue<T> values)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;


    if (row >= dst_info.h || column >= dst_info.w || channel >= dst_info.c)
        return;

    int q = channel - front;
    q = abs(q);
    q = (src_info.c - 1) - abs(q - (src_info.c - 1));

    const int dst_channel_step = dst_info.cstep * channel;
    const int src_channel_step = src_info.cstep * q;

    T* outptr = dst + dst_channel_step + row * dst_info.w + column;

    T padding_value{};
    if (values.per_channel_pad_data_size)
        padding_value = values.per_channel_values[channel];
    else
        padding_value = values.value;

    T output_value = padding_value;

    if (row < top) {
        const T* inptr = src + src_channel_step + (top - row) * src_info.w;
        if (column < left )
            output_value = inptr[left - column];
        else if (column >= left && column < left + src_info.w)
            output_value = inptr[column - left];
        else if (column < dst_info.w)
            output_value = inptr[src_info.w - (column - left - src_info.w) - 2];
    }
    else if (row >= top && row < top + src_info.h)
    {
        const T* inptr = src + src_channel_step + (row - top) * src_info.w;
        if (column < left)
            output_value = inptr[left - column];
        else if (column >= left+src_info.w)
            output_value = inptr[src_info.w - (column - left - src_info.w) - 2];
        else
            output_value = inptr[column - left];
    }
    else if (row >= top + src_info.h) {
        int diff = dst_info.h - top - src_info.h;
        const T* inptr = src + src_channel_step + (src_info.h - (diff - (dst_info.h-row)) - 2) * src_info.w;
        if (column < left)
            output_value = inptr[left - column];
        else if (column >= left + src_info.w)
            output_value = inptr[src_info.w - (column - left - src_info.w) - 2];
        else
            output_value = inptr[column - left];
    }

    *outptr = output_value;
}

int copy_make_border_image(const CudaMat& src, CudaMat& dst, int top, int left, int type, PaddingValue value, PaddingVariableType padding_type)
{
    const ncnn::CudaMatInfo input_info{src};
    const ncnn::CudaMatInfo output_info{dst};

    int thread_per_block_x = output_info.w;
    if (thread_per_block_x > 32) thread_per_block_x = 32;
    int thread_per_block_y = output_info.h;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    int thread_per_block_z = output_info.c;
    if (thread_per_block_z > 4) thread_per_block_z = 4;
    const int total_number_of_columns = output_info.w;
    const int total_number_of_rows = output_info.h;
    const int total_number_of_channels = output_info.c;
    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    if (type == 0)
    {
        if (padding_type == PaddingVariableType::type_char)
        {
            char value_char = value.c;
            gpu_copy_make_border_image_type0<char><<<grid_size, block_size>>>(static_cast<const char*>(src.get_craw_data()),
                                                                                input_info,
                                                                                static_cast<char*>(dst.get_raw_data()),
                                                                                output_info,
                                                                                top, left, type, value_char);
        }
        else if (padding_type == PaddingVariableType::type_unsigned_short)
        {
            unsigned short value_unsigned_short = value.sh;
            gpu_copy_make_border_image_type0<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(src.get_craw_data()),
                                                                                          input_info,
                                                                                          static_cast<unsigned short*>(dst.get_raw_data()),
                                                                                          output_info,
                                                                                          top, left, type, value_unsigned_short);
        }
        else if (padding_type == PaddingVariableType::type_float)
        {
            float value_float = value.fl;
            gpu_copy_make_border_image_type0<float><<<grid_size, block_size>>>(static_cast<const float*>(src.get_craw_data()),
                                                                                 input_info,
                                                                                 static_cast<float*>(dst.get_raw_data()),
                                                                                 output_info,
                                                                                 top, left, type, value_float);
        }
    }
    else if (type == 1)
    {
        if (padding_type == PaddingVariableType::type_char)
        {
            char value_char = value.c;
            gpu_copy_make_border_image_type1<char><<<grid_size, block_size>>>(static_cast<const char*>(src.get_craw_data()),
                                                                                input_info,
                                                                                static_cast<char*>(dst.get_raw_data()),
                                                                                output_info,
                                                                                top, left, type, value_char);
        }
        else if (padding_type == PaddingVariableType::type_unsigned_short)
        {
            unsigned short value_unsigned_short = value.sh;
            gpu_copy_make_border_image_type1<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(src.get_craw_data()),
                                                                                          input_info,
                                                                                          static_cast<unsigned short*>(dst.get_raw_data()),
                                                                                          output_info,
                                                                                          top, left, type, value_unsigned_short);
        }
        else if (padding_type == PaddingVariableType::type_float)
        {
            float value_float = value.fl;
            gpu_copy_make_border_image_type1<float><<<grid_size, block_size>>>(static_cast<const float*>(src.get_craw_data()),
                                                                                 input_info,
                                                                                 static_cast<float*>(dst.get_raw_data()),
                                                                                 output_info,
                                                                                 top, left, type, value_float);
        }
    }
    else if (type == 2)
    {
        if (padding_type == PaddingVariableType::type_char)
        {
            char value_char = value.c;
            gpu_copy_make_border_image_type2<char><<<grid_size, block_size>>>(static_cast<const char*>(src.get_craw_data()),
                                                                                input_info,
                                                                                static_cast<char*>(dst.get_raw_data()),
                                                                                output_info,
                                                                                top, left, type, value_char);
        }
        else if (padding_type == PaddingVariableType::type_unsigned_short)
        {
            unsigned short value_unsigned_short = value.sh;
            gpu_copy_make_border_image_type2<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(src.get_craw_data()),
                                                                                          input_info,
                                                                                          static_cast<unsigned short*>(dst.get_raw_data()),
                                                                                          output_info,
                                                                                          top, left, type, value_unsigned_short);
        }
        else if (padding_type == PaddingVariableType::type_float)
        {
            float value_float = value.fl;
            gpu_copy_make_border_image_type2<float><<<grid_size, block_size>>>(static_cast<const float*>(src.get_craw_data()),
                                                                                 input_info,
                                                                                 static_cast<float*>(dst.get_raw_data()),
                                                                                 output_info,
                                                                                 top, left, type, value_float);
        }
    }

    return 0;
}

int copy_make_border_image_3d(const CudaMat& src, CudaMat& dst, int front, int top, int left, int type,
                              PaddingValue value, PaddingVariableType padding_type,
                              void *gpu_per_channel_padding_data, int per_channel_pad_data_size)
{
    const ncnn::CudaMatInfo input_info{src};
    const ncnn::CudaMatInfo output_info{dst};

    int thread_per_block_x = output_info.w;
    if (thread_per_block_x > 32) thread_per_block_x = 32;
    int thread_per_block_y = output_info.h;
    if (thread_per_block_y > 8) thread_per_block_y = 8;
    int thread_per_block_z = output_info.c;
    if (thread_per_block_z > 4) thread_per_block_z = 4;
    const int total_number_of_columns = output_info.w;
    const int total_number_of_rows = output_info.h;
    const int total_number_of_channels = output_info.c;
    const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                         (total_number_of_rows - 1) / thread_per_block_y + 1,
                         (total_number_of_channels - 1) / thread_per_block_z + 1);

    if (type == 0)
    {
        if (padding_type == PaddingVariableType::type_char)
        {
            ncnn::GPUPaddingValue<char> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.c;
            padding_values.per_channel_values = static_cast<char*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type0<char><<<grid_size, block_size>>>(static_cast<const char*>(src.get_craw_data()),
                                                                                   input_info,
                                                                                   static_cast<char*>(dst.get_raw_data()),
                                                                                   output_info,
                                                                                   front, top, left, type,
                                                                                   padding_values);
        }
        else if (padding_type == PaddingVariableType::type_unsigned_short)
        {
            ncnn::GPUPaddingValue<unsigned short> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.sh;
            padding_values.per_channel_values = static_cast<unsigned short*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type0<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(src.get_craw_data()),
                                                                                             input_info,
                                                                                             static_cast<unsigned short*>(dst.get_raw_data()),
                                                                                             output_info,
                                                                                             front, top, left, type, padding_values);
        }
        else if (padding_type == PaddingVariableType::type_float)
        {
            ncnn::GPUPaddingValue<float> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.fl;
            padding_values.per_channel_values = static_cast<float*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type0<float><<<grid_size, block_size>>>(static_cast<const float*>(src.get_craw_data()),
                                                                                    input_info,
                                                                                    static_cast<float*>(dst.get_raw_data()),
                                                                                    output_info,
                                                                                    front, top, left, type, padding_values);
        }
    }
    else if (type == 1)
    {
        if (padding_type == PaddingVariableType::type_char)
        {
            ncnn::GPUPaddingValue<char> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.c;
            padding_values.per_channel_values = static_cast<char*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type1<char><<<grid_size, block_size>>>(static_cast<const char*>(src.get_craw_data()),
                                                                                   input_info,
                                                                                   static_cast<char*>(dst.get_raw_data()),
                                                                                   output_info,
                                                                                   front, top, left, type,
                                                                                   padding_values);
        }
        else if (padding_type == PaddingVariableType::type_unsigned_short)
        {
            ncnn::GPUPaddingValue<unsigned short> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.sh;
            padding_values.per_channel_values = static_cast<unsigned short*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type1<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(src.get_craw_data()),
                                                                                             input_info,
                                                                                             static_cast<unsigned short*>(dst.get_raw_data()),
                                                                                             output_info,
                                                                                             front, top, left, type, padding_values);
        }
        else if (padding_type == PaddingVariableType::type_float)
        {
            ncnn::GPUPaddingValue<float> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.fl;
            padding_values.per_channel_values = static_cast<float*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type1<float><<<grid_size, block_size>>>(static_cast<const float*>(src.get_craw_data()),
                                                                                    input_info,
                                                                                    static_cast<float*>(dst.get_raw_data()),
                                                                                    output_info,
                                                                                    front, top, left, type, padding_values);
        }
    }
    else if (type == 2)
    {
        if (padding_type == PaddingVariableType::type_char)
        {
            ncnn::GPUPaddingValue<char> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.c;
            padding_values.per_channel_values = static_cast<char*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type2<char><<<grid_size, block_size>>>(static_cast<const char*>(src.get_craw_data()),
                                                                                   input_info,
                                                                                   static_cast<char*>(dst.get_raw_data()),
                                                                                   output_info,
                                                                                   front, top, left, type,
                                                                                   padding_values);
        }
        else if (padding_type == PaddingVariableType::type_unsigned_short)
        {
            ncnn::GPUPaddingValue<unsigned short> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.sh;
            padding_values.per_channel_values = static_cast<unsigned short*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type2<unsigned short><<<grid_size, block_size>>>(static_cast<const unsigned short*>(src.get_craw_data()),
                                                                                             input_info,
                                                                                             static_cast<unsigned short*>(dst.get_raw_data()),
                                                                                             output_info,
                                                                                             front, top, left, type, padding_values);
        }
        else if (padding_type == PaddingVariableType::type_float)
        {
            ncnn::GPUPaddingValue<float> padding_values{};
            padding_values.per_channel_pad_data_size = per_channel_pad_data_size;
            padding_values.value = value.fl;
            padding_values.per_channel_values = static_cast<float*>(gpu_per_channel_padding_data);
            gpu_copy_make_border_image_3d_type2<float><<<grid_size, block_size>>>(static_cast<const float*>(src.get_craw_data()),
                                                                                    input_info,
                                                                                    static_cast<float*>(dst.get_raw_data()),
                                                                                    output_info,
                                                                                    front, top, left, type, padding_values);
        }
    }

    return 0;
}

}
