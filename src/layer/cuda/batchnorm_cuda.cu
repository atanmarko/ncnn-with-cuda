#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_util.h"

#include <iostream>

__global__ void gpu_batchnorm_load_model(int channels, float eps, float* a_data_gpu, float* b_data_gpu,
                                         float* bias_data_gpu, float* slope_data_gpu, float* mean_data_gpu, float* var_data_gpu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    const float sqrt_var = static_cast<float>(sqrt(var_data_gpu[i] + eps));
    a_data_gpu[i] = bias_data_gpu[i] - slope_data_gpu[i] * mean_data_gpu[i] / sqrt_var;
    b_data_gpu[i] = slope_data_gpu[i] / sqrt_var;
}

__global__ void gpu_batchnorm_forward_inplace(float* d_input, const int input_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x; //limited to 1024 rows

    if (index >= input_size) return;

    d_input[index] = d_input[index] >= 0 ? d_input[index] : -d_input[index];
}


namespace ncnn {


int batchnorm_load_model(int channels, float eps, float* a_data_gpu, float* b_data_gpu,
                         float* bias_data_gpu, float* slope_data_gpu, float* mean_data_gpu, float* var_data_gpu)
{

    gpu_batchnorm_load_model<<<1, channels>>>(channels, eps, a_data_gpu, b_data_gpu,
                                                bias_data_gpu, slope_data_gpu, mean_data_gpu,
                                                var_data_gpu);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return 0;
}




int relu_batchnorm_forward_inplace(float* d_input, const int input_size)
{
    const int thread_per_block = 512;
    const dim3 block_size(thread_per_block, 1, 1);
    const dim3 grid_size(input_size / thread_per_block + 1, 1, 1);

    gpu_batchnorm_forward_inplace<<<grid_size, block_size>>>(d_input, input_size);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return 0;
}
} // namespace ncnn
