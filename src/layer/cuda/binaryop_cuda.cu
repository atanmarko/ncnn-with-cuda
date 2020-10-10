//
// Copyright (C) 2020 TANCOM SOFTWARE SOLUTIONS LLC. All rights reserved.
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
#include "mat.h"

#include <binaryop.h>

#include <iostream>


namespace ncnn {

struct binary_op_add_cuda
{
    __device__ binary_op_add_cuda(){};
    __device__ ~binary_op_add_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return x + y;
    }
};

struct binary_op_sub_cuda
{
    __device__ binary_op_sub_cuda(){};
    __device__ ~binary_op_sub_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return x - y;
    }
};

struct binary_op_mul_cuda
{
    __device__ binary_op_mul_cuda(){};
    __device__ ~binary_op_mul_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return x * y;
    }
};

struct binary_op_div_cuda
{
    __device__ binary_op_div_cuda(){};
    __device__ ~binary_op_div_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return x / y;
    }
};

struct binary_op_max_cuda
{
    __device__ binary_op_max_cuda(){};
    __device__ ~binary_op_max_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        if (x < y)
            return y;
        else
            return x;
    }
};

struct binary_op_min_cuda
{
    __device__ binary_op_min_cuda(){};
    __device__ ~binary_op_min_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        if (x < y)
            return x;
        else
            return y;
    }
};

struct binary_op_pow_cuda
{
    __device__ binary_op_pow_cuda(){};
    __device__ ~binary_op_pow_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return (float)pow(x, y);
    }
};

struct binary_op_rsub_cuda
{
    __device__ binary_op_rsub_cuda(){};
    __device__ ~binary_op_rsub_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return y - x;
    }
};

struct binary_op_rdiv_cuda
{
    __device__ binary_op_rdiv_cuda(){};
    __device__ ~binary_op_rdiv_cuda(){};
    __device__ float operator()(const float& x, const float& y) const
    {
        return y / x;
    }
};

// input is 1 dimension
template<typename Op>
__global__ void gpu_binaryop_forward_inplace(float* a_input, const ncnn::CudaMatInfo a_info, const float b)
{
    const int input_size = a_info.cstep * a_info.c;
    if (blockIdx.x * blockDim.x + threadIdx.x >= input_size) return;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int channelSize = a_info.cstep;
    const int channel = index / channelSize;
    const int row = (index - (channel * channelSize)) / a_info.w;
    const int column = (index - (channel * channelSize)) % a_info.w;
    const int channel_step = channel * channelSize * a_info.elemsize;

    float* ptr = (float*)((unsigned char*)a_input + channel_step);
    const int i = row * a_info.w+column;

    Op op;

    ptr[i] = op(ptr[i], b);
}

template<typename Op>
__global__ void gpu_binaryop_forward_1(const float* a_input, const ncnn::CudaMatInfo a_info, const float* b_input,
                                        const ncnn::CudaMatInfo b_info, float* out, const ncnn::CudaMatInfo out_info)
{
    // If a_input or b_input have width or height 1
    const int a_input_size = a_info.cstep * a_info.c;
    const int b_input_size = b_info.cstep * b_info.c;
    const int input_size = a_input_size >= b_input_size ? a_input_size : b_input_size;
    const int output_size = out_info.cstep * out_info.c;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size || index >= output_size) return;

    Op op;

    if (a_info.w == 1) {
        const float a0 = a_input[0];
        out[index] = op(a0, b_input[index]);
    }
    else if (b_info.w == 1)
    {
        const float b0 = *(float*)((unsigned char*)b_input);
        out[index] = op(a_input[index], b0);
    }
    else {
        out[index] = op(a_input[index], b_input[index]);
    }

}

template<typename Op>
__global__ void gpu_binaryop_forward_12(const float* a_input, const ncnn::CudaMatInfo a_info, const float* b_input,
                                       const ncnn::CudaMatInfo b_info, float* out, const ncnn::CudaMatInfo out_info)
{
    // If a_input or b_input have width or height 1
    const int a_input_size = a_info.cstep * a_info.c;
    const int b_input_size = b_info.cstep * b_info.c;
    const int input_size = a_input_size >= b_input_size ? a_input_size : b_input_size;
    const int output_size = out_info.cstep * out_info.c;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size || index >= output_size) return;

    Op op;


    if (a_info.dims == 1 && a_info.w !=1) {
        const int row = index / b_info.w;
        const int column = index % b_info.w;
        const float a0 = a_input[row];
        out[index] = op(a0, b_input[row*b_info.w+column]);
    } else if (b_info.dims == 1 && b_info.w !=1) {
        const int row = index / a_info.w;
        const int column = index % a_info.w;
        const float b0 = b_input[row];
        out[index] = op(a_input[row*a_info.w+column], b0);

    }
}

template<typename Op>
__global__ void gpu_binaryop_forward_13(const float* a_input, const ncnn::CudaMatInfo a_info, const float* b_input,
                                        const ncnn::CudaMatInfo b_info, float* out, const ncnn::CudaMatInfo out_info)
{
    // If a_input or b_input have width or height 1
    const int a_input_size = a_info.cstep * a_info.c;
    const int b_input_size = b_info.cstep * b_info.c;
    const int input_size = a_input_size >= b_input_size ? a_input_size : b_input_size;
    const int output_size = out_info.cstep * out_info.c;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size || index >= output_size) return;

    Op op;

    const int out_channelSize = out_info.cstep;
    const int channel = index / out_channelSize;
    const int row = (index - (channel * out_channelSize)) / out_info.w;
    const int column = (index - (channel * out_channelSize)) % out_info.w;
    const int channel_step = channel * out_channelSize * out_info.elemsize;

    if (a_info.dims == 1 && b_info.dims == 3)
    {
        const int i = channel * out_channelSize + row * b_info.w + column;
        const float a0 = a_input[channel];
        out[i] = op(a0, b_input[i]);
    } else if (a_info.dims == 3 && b_info.dims == 1) {
        const int i = channel * out_channelSize + row * a_info.w + column;
        const float b0 = b_input[channel];
        out[i] = op(a_input[i], b0);
    }
}


template<typename Op>
__global__ void gpu_binaryop_forward_31(const float* a_input, const ncnn::CudaMatInfo a_info, const float* b_input,
                                        const ncnn::CudaMatInfo b_info, float* out, const ncnn::CudaMatInfo out_info)
{
    // If a_input or b_input have width or height 1
    const int a_input_size = a_info.cstep * a_info.c;
    const int b_input_size = b_info.cstep * b_info.c;
    const int input_size = a_input_size >= b_input_size ? a_input_size : b_input_size;
    const int output_size = out_info.cstep * out_info.c;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size || index >= output_size) return;

    const int out_channelSize = out_info.cstep;
    const int channel = index / out_channelSize;
    const int row = (index - (channel * out_channelSize)) / out_info.w;
    const int column = (index - (channel * out_channelSize)) % out_info.w;

    const int i = row * out_info.w+ column;

    Op op;

    if (a_info.w == 1 && a_info.h == 1) {
        const float a0 = a_input[channel*a_info.cstep];
        out[channel*out_info.cstep+i] = op(a0, b_input[channel*b_info.cstep+i]);
    }
    else if (b_info.w == 1 && b_info.h == 1)
    {
        float b0{0};
        if (b_info.c == 1)
            b0 = b_input[0];
        else
            b0 = b_input[channel*b_info.cstep];

        out[channel*out_info.cstep+i] = op(a_input[channel*a_info.cstep+i], b0);
    }
}
template<typename Op>
__global__ void gpu_binaryop_forward_32(const float* a_input, const ncnn::CudaMatInfo a_info, const float* b_input,
                                        const ncnn::CudaMatInfo b_info, float* out, const ncnn::CudaMatInfo out_info)
{
    // If a_input or b_input have width or height 1
    const int a_input_size = a_info.cstep * a_info.c;
    const int b_input_size = b_info.cstep * b_info.c;
    const int input_size = a_input_size >= b_input_size ? a_input_size : b_input_size;
    const int output_size = out_info.cstep * out_info.c;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size || index >= output_size) return;

    const int out_channelSize = out_info.cstep;
    const int channel = index / out_channelSize;
    const int row = (index - (channel * out_channelSize)) / out_info.w;
    const int column = (index - (channel * out_channelSize)) % out_info.w;
    const int channel_step = channel * out_channelSize;

    const int i = channel_step + row * out_info.w + column;

    Op op;
    if (a_info.dims == 2 && b_info.dims == 3) {
        out[i] = op(a_input[channel*a_info.w+ row], b_input[i]);
    } else if (a_info.dims == 3 && b_info.dims == 2) {
        out[i] = op(a_input[i], b_input[channel*b_info.w+ row]);
    }


}

template<typename Op>
__global__ void gpu_binaryop_forward_33(const float* a_input, const ncnn::CudaMatInfo a_info, const float* b_input,
                                        const ncnn::CudaMatInfo b_info, float* out, const ncnn::CudaMatInfo out_info)
{
    // If a_input or b_input have width or height 1
    const int a_input_size = a_info.cstep * a_info.c;
    const int b_input_size = b_info.cstep * b_info.c;
    const int input_size = a_input_size >= b_input_size ? a_input_size : b_input_size;
    const int output_size = out_info.cstep * out_info.c;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= input_size || index >= output_size) return;

    const int out_channelSize = out_info.cstep;
    const int channel = index / out_channelSize;
    const int row = (index - (channel * out_channelSize)) / out_info.w;
    const int column = (index - (channel * out_channelSize)) % out_info.w;
    const int channel_step = channel * out_channelSize;

    const int i = channel_step + row * out_info.w+ column;

    Op op;

    if (b_info.c == 1) {
        out[i] = op(a_input[i], b_input[row * b_info.w + column]);
    }
    else if (a_info.c == 1) {
        out[i] = op(a_input[row * a_info.w + column], b_input[i]);
    }
    else {
        out[i] = op(a_input[i], b_input[i]);
    }
}


template<typename Op>
static int binary_op_cuda(const float* a_input, const CudaMatInfo& a_info, const float* b_input,
                          const CudaMatInfo& b_info, float* c_output, const CudaMatInfo& c_info)
{
    const int channels_a = a_info.c;
    const int channels_b = b_info.c;

    const int num_matrix_elements = std::max(a_info.c*a_info.cstep, b_info.c*b_info.cstep);
    int thread_per_block = 512 > num_matrix_elements ? 512 : ((num_matrix_elements / 32) + 1) * 32;
    if (thread_per_block > 1024) thread_per_block = 1024;

    if (a_info.dims == 3)
    {
        if (b_info.dims == 3) {
            if (a_info.w == 1 && a_info.h == 1 && channels_a == channels_b)
            {
                const int input_size = b_info.cstep * b_info.c;
                const dim3 block_size(thread_per_block, 1, 1);
                const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
                gpu_binaryop_forward_31<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
            }
            else if (b_info.w == 1 && b_info.h == 1 && channels_a == channels_b)
            {
                const int input_size = a_info.cstep * a_info.c;
                const dim3 block_size(thread_per_block, 1, 1);
                const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
                gpu_binaryop_forward_31<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
            }
            else {
                const int input_size = a_info.cstep * a_info.c > b_info.cstep * b_info.c ? a_info.cstep * a_info.c : b_info.cstep * b_info.c;
                const dim3 block_size(thread_per_block, 1, 1);
                const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
                gpu_binaryop_forward_33<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);

            }
        } else if (b_info.dims == 2) {
            const int input_size = a_info.cstep * a_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            gpu_binaryop_forward_32<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
        else if (b_info.dims == 1) {
            const int input_size = a_info.cstep * a_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            if (b_info.w == 1)
                gpu_binaryop_forward_1<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
            else
                gpu_binaryop_forward_13<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
    }
    else if (a_info.dims == 2)
    {
        if (b_info.dims == 1)
        {
            const int input_size = a_info.cstep * a_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);

            if (b_info.w == 1)
                gpu_binaryop_forward_1<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
            else
                gpu_binaryop_forward_12<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);

        } else if (b_info.dims == 2) {
            const int input_size = a_info.cstep * a_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            gpu_binaryop_forward_33<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }  else if (b_info.dims == 3) {
            const int input_size = b_info.cstep * b_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            gpu_binaryop_forward_32<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
    }
    else if (a_info.dims == 1) {

        if (a_info.w == 1) {
            const int input_size = b_info.cstep * b_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            gpu_binaryop_forward_1<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
        else if (b_info.dims == 1) {
                const int input_size = a_info.cstep * a_info.c;
                const dim3 block_size(thread_per_block, 1, 1);
                const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
                gpu_binaryop_forward_1<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
        else if (b_info.dims == 2) {
            const int input_size = b_info.cstep * b_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            gpu_binaryop_forward_12<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
        else if (b_info.dims == 3) {
            const int input_size = b_info.cstep * b_info.c;
            const dim3 block_size(thread_per_block, 1, 1);
            const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);
            gpu_binaryop_forward_13<Op><<<grid_size, block_size>>>(a_input, a_info, b_input, b_info, c_output, c_info);
        }
    }


    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return 0;
}

template<typename Op>
static int binaryop_scalar_cuda_inplace(float* a_input, const CudaMatInfo& a_info, float b)
{
    const int thread_per_block = 512;
    const int input_size = a_info.cstep * a_info.c;

    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);

    gpu_binaryop_forward_inplace<Op><<<grid_size, block_size>>>(a_input, a_info, b);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return 0;
}



int binary_op_cuda_forward(const CudaMat& a, const CudaMat& b, CudaMat& c, const Option& opt, int op_type)
{
    const float* a_gpu = static_cast<const float*>(a.get_craw_data());
    const float* b_gpu = static_cast<const float*>(b.get_craw_data());
    CudaMatInfo a_info{a};
    CudaMatInfo b_info{b};

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    if (a_info.dims == 3) {
        if ((a_info.w == 1 && a_info.h == 1 && a_info.c == b_info.c) ||
            (a_info.w == b_info.w && a_info.h == b_info.h && a_info.c == 1))
        {
            c.create(b_info.w, b_info.h, b_info.c, b_info.elemsize, cuda_allocator);
        }
        else {
            c.create(a_info.w, a_info.h, a_info.c, a_info.elemsize, cuda_allocator);
        }
    } else if (a_info.dims == 2) {
        if (b_info.dims == 3) {
            c.create(b_info.w, b_info.h, b_info.c, a_info.elemsize, cuda_allocator);
        } else {
            c.create(a_info.w, a_info.h, a_info.elemsize, cuda_allocator);
        }
    } else if (a_info.dims == 1) {
            if (b.dims == 3)
                c.create(b_info.w, b_info.h, b_info.c, b_info.elemsize, cuda_allocator);
            else if (b.dims == 2)
                c.create(b_info.w, b_info.h, b_info.elemsize, cuda_allocator);
            else if (b.dims == 1 ) {
                if (a_info.w != 1)
                    c.create(a_info.w, a_info.elemsize, cuda_allocator);
                else
                    c.create(b_info.w, b_info.elemsize, cuda_allocator);
            }
    }

    float* c_gpu = static_cast<float*>(c.get_raw_data());
    CudaMatInfo c_info{c};


    if (op_type == BinaryOp::Operation_ADD)
        return binary_op_cuda<binary_op_add_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_SUB)
        return binary_op_cuda<binary_op_sub_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_MUL)
        return binary_op_cuda<binary_op_mul_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_DIV)
        return binary_op_cuda<binary_op_div_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_MAX)
        return binary_op_cuda<binary_op_max_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_MIN)
        return binary_op_cuda<binary_op_min_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_POW)
        return binary_op_cuda<binary_op_pow_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_RSUB)
        return binary_op_cuda<binary_op_rsub_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    if (op_type == BinaryOp::Operation_RDIV)
        return binary_op_cuda<binary_op_rdiv_cuda>(a_gpu, a_info, b_gpu, b_info, c_gpu, c_info);

    return 0;
}

int binary_op_scalar_inplace_cuda_forward(CudaMat& a, float b, const Option& opt, int op_type)
{
    float* a_gpu = static_cast<float*>(a.get_raw_data());
    CudaMatInfo a_info{a};

    if (op_type == BinaryOp::Operation_ADD)
        return binaryop_scalar_cuda_inplace<binary_op_add_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_SUB)
        return binaryop_scalar_cuda_inplace<binary_op_sub_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_MUL)
        return binaryop_scalar_cuda_inplace<binary_op_mul_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_DIV)
        return binaryop_scalar_cuda_inplace<binary_op_div_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_MAX)
        return binaryop_scalar_cuda_inplace<binary_op_max_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_MIN)
        return binaryop_scalar_cuda_inplace<binary_op_min_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_POW)
        return binaryop_scalar_cuda_inplace<binary_op_pow_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_RSUB)
        return binaryop_scalar_cuda_inplace<binary_op_rsub_cuda>(a_gpu, a_info, b);

    if (op_type == BinaryOp::Operation_RDIV)
        return binaryop_scalar_cuda_inplace<binary_op_rdiv_cuda>(a_gpu, a_info, b);

    return 0;
}

} // namespace ncnn
