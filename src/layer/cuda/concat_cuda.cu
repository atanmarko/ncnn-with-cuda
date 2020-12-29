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


__global__ void gpu_concat_forward_dim1(void* inputs, const ncnn::CudaMatInfo* input_info,
                                   int* input_offset, const int input_size,
                                   unsigned char* output, const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column >= output_info.w)
    {
        return;
    }

    //find input for this output
    int input_matrix_index = 0;
    int i=0;
    while (input_offset[i] <= column && i < input_size) {
        input_matrix_index = i;
        ++i;
    }

    const  unsigned char* current_input = reinterpret_cast<const unsigned char*>(*(reinterpret_cast<const long long *>
                                                                                  (static_cast<const unsigned char*>(inputs)+input_matrix_index*sizeof(const char *))));
    const ncnn::CudaMatInfo& current_input_info  = input_info[input_matrix_index];

    const int input_index = (column - input_offset[input_matrix_index]) * input_info[input_matrix_index].elemsize;
    const int output_index = column * output_info.elemsize;
    memcpy((void*)(output + output_index), (void*)(current_input + input_index), current_input_info.elemsize);

}

__global__ void gpu_concat_forward_dim3_height(void* inputs, const ncnn::CudaMatInfo* input_info,
                                        int* input_offset, const int input_size,
                                        unsigned char* output, const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= output_info.w || row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    //find input for this output
    int input_matrix_index = 0;
    int i=0;
    while (input_offset[i] <= row && i < input_size) {
        input_matrix_index = i;
        ++i;
    }

    const  unsigned char* current_input = reinterpret_cast<const unsigned char*>(*(reinterpret_cast<const long long *>
    (static_cast<const unsigned char*>(inputs)+ input_matrix_index * sizeof(const char *))));
    const ncnn::CudaMatInfo& current_input_info  = input_info[input_matrix_index];

    const int input_index = channel * input_info[input_matrix_index].cstep * input_info[input_matrix_index].elemsize
                            + (row - input_offset[input_matrix_index]) * input_info[input_matrix_index].w * output_info.elemsize
                            + column * input_info[input_matrix_index].elemsize;
    const int output_index = channel * output_info.cstep * output_info.elemsize + row * output_info.w * output_info.elemsize + column * output_info.elemsize;
    memcpy((void*)(output + output_index), (void*)(current_input + input_index), current_input_info.elemsize);
}

__global__ void gpu_concat_forward_dim3_width(void* inputs, const ncnn::CudaMatInfo* input_info,
                                        int* input_offset, const int input_size,
                                        unsigned char* output, const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= output_info.w || row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    //find input for this output
    int input_matrix_index = 0;
    int i=0;
    while (input_offset[i] <= column && i < input_size) {
        input_matrix_index = i;
        ++i;
    }

    const  unsigned char* current_input = reinterpret_cast<const unsigned char*>(*(reinterpret_cast<const long long *>
    (static_cast<const unsigned char*>(inputs)+ input_matrix_index * sizeof(const char *))));
    const ncnn::CudaMatInfo& current_input_info  = input_info[input_matrix_index];

    const int input_index = channel * input_info[input_matrix_index].cstep * input_info[input_matrix_index].elemsize
                            + row * input_info[input_matrix_index].w * output_info.elemsize
                            + (column - input_offset[input_matrix_index]) * input_info[input_matrix_index].elemsize;
    const int output_index = channel * output_info.cstep * output_info.elemsize + row * output_info.w * output_info.elemsize + column * output_info.elemsize;
    memcpy((void*)(output + output_index), (void*)(current_input + input_index), current_input_info.elemsize);
}


__global__ void gpu_concat_forward_dim3_channel(void* inputs, const ncnn::CudaMatInfo* input_info,
                                              int* input_offset, const int input_size,
                                              unsigned char* output, const ncnn::CudaMatInfo output_info) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (column >= output_info.w || row >= output_info.h || channel >= output_info.c)
    {
        return;
    }

    //find input for this output
    int input_matrix_index = 0;
    int i=0;
    while (input_offset[i] <= channel && i < input_size) {
        input_matrix_index = i;
        ++i;
    }

    const  unsigned char* current_input = reinterpret_cast<const unsigned char*>(*(reinterpret_cast<const long long *>
    (static_cast<const unsigned char*>(inputs)+ input_matrix_index * sizeof(const char *))));
    const ncnn::CudaMatInfo& current_input_info  = input_info[input_matrix_index];

    const int input_index = (channel - input_offset[input_matrix_index]) * input_info[input_matrix_index].cstep * input_info[input_matrix_index].elemsize
                            + row * input_info[input_matrix_index].w * output_info.elemsize
                            + column * input_info[input_matrix_index].elemsize;
    const int output_index = channel * output_info.cstep * output_info.elemsize + row * output_info.w * output_info.elemsize + column * output_info.elemsize;
    memcpy((void*)(output + output_index), (void*)(current_input + input_index), current_input_info.elemsize);
}



namespace ncnn {

int concat_cuda_forward(const std::vector<CudaMat>& bottom_blobs, CudaMat& top_blob, const int axis)
{
    const int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;
    const int positive_axis = axis < 0 ? dims + axis : axis;
    const int input_size = (bottom_blobs.size());



    std::vector<int> input_offset;
    std::vector<ncnn::CudaMatInfo> input_info;
    std::vector<const int *> inputs;
    int* gpu_input_offset;
    ncnn::CudaMatInfo* gpu_input_info;
    int* gpu_inputs;

    const ncnn::CudaMatInfo output_info{top_blob};

    checkCudaErrors(cudaMalloc(&gpu_input_offset, input_size*sizeof(int)));
    checkCudaErrors(cudaMalloc(&gpu_input_info, input_size*sizeof(ncnn::CudaMatInfo)));
    checkCudaErrors(cudaMalloc(&gpu_inputs, input_size * sizeof(int*)));


    if (dims == 1) {
        int thread_per_block_x = ((top_blob.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = 1;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = top_blob.c;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        //find input offsets
        int offset = 0;
        for (int i=0; i<input_size; ++i) {
            input_offset.push_back(offset);
            input_info.push_back(ncnn::CudaMatInfo{bottom_blobs[i]});
            inputs.push_back(static_cast<const int*>(bottom_blobs[i].get_craw_data()));
            offset += bottom_blobs[i].w;
        }
        checkCudaErrors(cudaMemcpy(gpu_input_offset, input_offset.data(), input_offset.size()*sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_input_info, input_info.data(), input_info.size()*sizeof(ncnn::CudaMatInfo), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(int*), cudaMemcpyHostToDevice));

        gpu_concat_forward_dim1<<<grid_size, block_size>>>(gpu_inputs, gpu_input_info, gpu_input_offset, input_size,
                                                             static_cast<unsigned char*>(top_blob.get_raw_data()), output_info);
    }

    if ((dims == 2 && positive_axis == 0)
        || (dims == 3 && positive_axis == 1)) {

        int thread_per_block_x = ((top_blob.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = top_blob.c;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        //find input offsets
        int offset_h = 0;
        for (int i=0; i<input_size; ++i) {
            input_offset.push_back(offset_h);
            input_info.push_back(ncnn::CudaMatInfo{bottom_blobs[i]});
            inputs.push_back(static_cast<const int*>(bottom_blobs[i].get_craw_data()));
            offset_h += bottom_blobs[i].h;
        }
        checkCudaErrors(cudaMemcpy(gpu_input_offset, input_offset.data(), input_offset.size()*sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_input_info, input_info.data(), input_info.size()*sizeof(ncnn::CudaMatInfo), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(int*), cudaMemcpyHostToDevice));

        gpu_concat_forward_dim3_height<<<grid_size, block_size>>>(gpu_inputs, gpu_input_info, gpu_input_offset, input_size,
                                                                    static_cast<unsigned char*>(top_blob.get_raw_data()), output_info);

    }

    if ((dims == 2 && positive_axis == 1) ||
        (dims == 3 && positive_axis == 2)){

        int thread_per_block_x = ((top_blob.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = top_blob.c;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        //find input offsets
        int offset_w = 0;
        for (int i=0; i<input_size; ++i) {
            input_offset.push_back(offset_w);
            input_info.push_back(ncnn::CudaMatInfo{bottom_blobs[i]});
            inputs.push_back(static_cast<const int*>(bottom_blobs[i].get_craw_data()));
            offset_w += bottom_blobs[i].w;
        }
        checkCudaErrors(cudaMemcpy(gpu_input_offset, input_offset.data(), input_offset.size()*sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_input_info, input_info.data(), input_info.size()*sizeof(ncnn::CudaMatInfo), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(int*), cudaMemcpyHostToDevice));

        gpu_concat_forward_dim3_width<<<grid_size, block_size>>>(gpu_inputs, gpu_input_info, gpu_input_offset, input_size,
                                                                   static_cast<unsigned char*>(top_blob.get_raw_data()), output_info);

    }

    if (dims == 3 && positive_axis == 0) {

        int thread_per_block_x = ((top_blob.w - 1) / 64 + 1) * 64;
        if (thread_per_block_x > 128) thread_per_block_x = 128;
        int thread_per_block_y = 8;
        const int thread_per_block_z = 1;
        const int total_number_of_channels = top_blob.c;
        const int total_number_of_columns = top_blob.w;
        const int total_number_of_rows = top_blob.h;

        const dim3 block_size(thread_per_block_x, thread_per_block_y, thread_per_block_z);
        const dim3 grid_size((total_number_of_columns - 1) / thread_per_block_x + 1,
                             (total_number_of_rows - 1) / thread_per_block_y + 1,
                             (total_number_of_channels - 1) / thread_per_block_z + 1);

        //find input offsets
        int offset_c = 0;
        for (int i=0; i<input_size; ++i) {
            input_offset.push_back(offset_c);
            input_info.push_back(ncnn::CudaMatInfo{bottom_blobs[i]});
            inputs.push_back(static_cast<const int*>(bottom_blobs[i].get_craw_data()));
            offset_c += bottom_blobs[i].c;
        }
        checkCudaErrors(cudaMemcpy(gpu_input_offset, input_offset.data(), input_offset.size()*sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_input_info, input_info.data(), input_info.size()*sizeof(ncnn::CudaMatInfo), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(int*), cudaMemcpyHostToDevice));

        gpu_concat_forward_dim3_channel<<<grid_size, block_size>>>(gpu_inputs, gpu_input_info, gpu_input_offset, input_size,
                                                                 static_cast<unsigned char*>(top_blob.get_raw_data()), output_info);

    }


    checkCudaErrors(cudaFree(gpu_input_offset));
    checkCudaErrors(cudaFree(gpu_input_info));
    checkCudaErrors(cudaFree(gpu_inputs));

    //todo other dims

    return 0;
}


}