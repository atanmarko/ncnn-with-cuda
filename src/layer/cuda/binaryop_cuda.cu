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

    const int channelSize = a_info.cstep;
    const int channel = (blockIdx.x * blockDim.x + threadIdx.x) / channelSize;
    const int row = ((blockIdx.x * blockDim.x + threadIdx.x) - (channel * channelSize)) / a_info.w;
    const int column = ((blockIdx.x * blockDim.x + threadIdx.x) - (channel * channelSize)) % a_info.w;
    const int channel_step = channel * channelSize * a_info.elemsize;

    float* ptr = (float*)((unsigned char*)a_input + channel_step);
    const int i = row * a_info.w+column;

    Op op;

    ptr[i] = op(ptr[i], b);
}


template<typename Op>
static int binary_op_cuda(const float* a_input, const CudaMatInfo& a_info, const float* b_input,
                          const CudaMatInfo& b_info, const float* c_output, const CudaMatInfo& c_info)
{
//    Op op;
//    const int thread_per_block = 512;
    const int channels_a = a_info.c;
    const int channels_b = b_info.c;

    if (a_info.dims == 3)
    {
        if (b_info.w == 1 && b_info.h == 1 && channels_a == channels_b)
        {


        }
//        const int total_input_size = matInfo.cstep * matInfo.c;
//        dim3 block_size(thread_per_block,1,1);
//        dim3 grid_size(total_input_size / thread_per_block + 1, 1, 1);
//        gpu_binaryop_forward_inplace_3<<<grid_size, block_size>>>(d_input, b_data, a_data, matInfo, total_input_size);
    }

//    if (matInfo.dims == 1)
//    {
//        const int input_size = matInfo.w;
//        dim3 block_size(thread_per_block,1,1);
//        dim3 grid_size(matInfo.w / thread_per_block + 1, 1, 1);
//        gpu_binaryop_forward_inplace_1<<<grid_size, block_size>>>(d_input, b_data, a_data, matInfo, input_size);
//    }
//    if (matInfo.dims == 2)
//    {
//        const int input_size = matInfo.w * matInfo.h;
//        dim3 block_size(thread_per_block,1,1);
//        dim3 grid_size( input_size / thread_per_block + 1, 1, 1);
//        gpu_binaryop_forward_inplace_2<<<grid_size, block_size>>>(d_input, b_data, a_data, matInfo, input_size);
//    }


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
    const float* c_gpu = static_cast<const float*>(c.get_raw_data());
    CudaMatInfo a_info{a};
    CudaMatInfo b_info{b};
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
