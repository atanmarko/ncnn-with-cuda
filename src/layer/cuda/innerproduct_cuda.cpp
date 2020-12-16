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


#include "layer_type.h"
#include "innerproduct_cuda.h"

#include <chrono>

namespace ncnn {

int innerproduct_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const InnerProduct_cuda::InnerProduct_info& info,
                              float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size);
int innerproduct_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const InnerProduct_cuda::InnerProduct_info& info,
                                   float* gpu_scratchpad_memory, int gpu_scratchpad_memory_size);

InnerProduct_cuda::InnerProduct_cuda()
{
    support_cuda = true;
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_scratch_pad_memory = static_cast<float *>(cuda_allocator->fastMalloc(gpu_scratch_pad_memory_size));
    gpu_bottom_blob_int8_scale = static_cast<float *>(cuda_allocator->fastMalloc(sizeof(float)));
}

InnerProduct_cuda::~InnerProduct_cuda()
{
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    cuda_allocator->fastFree(gpu_scratch_pad_memory);
    cuda_allocator->fastFree(gpu_bottom_blob_int8_scale);
}

int InnerProduct_cuda::load_param(const ParamDict& pd)
{
    InnerProduct::load_param(pd);

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_activation_params = CudaMat{activation_params, cuda_allocator};

    return 0;
}

int InnerProduct_cuda::load_model(const CudaModelBinFromMatArray& mb)
{
    int result = InnerProduct::load_model(static_cast<ModelBinFromMatArray>(mb));
    if (result < 0)
        return result;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    gpu_weight_data = CudaMat{weight_data, cuda_allocator};

    if (bias_term)
    {
        gpu_bias_data = CudaMat{bias_data, cuda_allocator};
    }

    if (int8_scale_term)
    {
        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
    }

    return 0;
}

int InnerProduct_cuda::create_pipeline(const Option& opt)
{
    const int weight_elemsize = weight_data.elemsize;
    InnerProduct::create_pipeline(opt);

    if (opt.use_int8_inference && weight_elemsize == (size_t)4u && int8_scale_term)
    {
        std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
        gpu_weight_data = CudaMat{weight_data, cuda_allocator};
    }

    return 0;
}

int InnerProduct_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
    if (opt.use_int8_inference && gpu_weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    top_blob.create(num_output, bottom_blob.elemsize, cuda_allocator);
    if (top_blob.empty())
        return -100;

    return innerproduct_cuda_forward(bottom_blob, top_blob, InnerProduct_info(*this), gpu_scratch_pad_memory, gpu_scratch_pad_memory_size);
}

int InnerProduct_cuda::forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
    CudaMat bottom_blob_tm = bottom_blob;
    if (bottom_blob.elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_tm, bottom_blob_int8_scale, opt_g);
    }

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    top_blob.create(num_output, 4u, cuda_allocator);
    if (top_blob.empty())
        return -100;

    return innerproduct_cuda_forward_int8(bottom_blob_tm, top_blob, InnerProduct_info(*this),
                                          gpu_scratch_pad_memory, gpu_scratch_pad_memory_size);
}


} // namespace ncnn
