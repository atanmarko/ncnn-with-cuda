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


#ifndef LAYER_CONVOLUTIONDEPTHWISE_CUDA_H
#define LAYER_CONVOLUTIONDEPTHWISE_CUDA_H

#include "convolutiondepthwise.h"

namespace ncnn {

class ConvolutionDepthWise_cuda : virtual public ConvolutionDepthWise
{
public:
    struct ConvolutionDepthWise_info
    {
        ConvolutionDepthWise_info(const ConvolutionDepthWise_cuda& info, std::vector<int>& _space_ofs, const Option& opt)
            : num_output(info.num_output),
              kernel_w(info.kernel_w),
              kernel_h(info.kernel_h),
              dilation_w(info.dilation_w),
              dilation_h(info.dilation_h),
              stride_w(info.stride_w),
              stride_h(info.stride_h),
              pad_left(info.pad_left),
              pad_right(info.pad_right),
              pad_top(info.pad_top),
              pad_bottom(info.pad_bottom),
              pad_value(info.pad_value),
              bias_term(info.bias_term),
              weight_data_size(info.weight_data_size),
              group(info.group),
              int8_scale_term(info.int8_scale_term),
              activation_type(info.activation_type),
              use_int8_requantize(info.use_int8_requantize),
              gpu_activation_params(&info.gpu_activation_params),
              gpu_weight_data(&info.gpu_weight_data),
              gpu_bias_data(&info.gpu_bias_data),
              gpu_weight_data_int8_scales(&info.gpu_weight_data_int8_scales),
              gpu_bottom_blob_int8_scales(&info.gpu_bottom_blob_int8_scales),
              gpu_top_blob_int8_scale(info.gpu_top_blob_int8_scale)

        {
            kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
            maxk = kernel_w * kernel_h;

            auto deleter = [&opt](int* pointer) {
              std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
              opt.blob_cuda_allocator->fastFree(pointer);
            };
            gpu_space_ofs = std::shared_ptr<int>(static_cast<int*>(opt.blob_cuda_allocator->fastMalloc(_space_ofs.size() * sizeof(int))), deleter);
            checkCudaErrors(cudaMemcpy(gpu_space_ofs.get(), _space_ofs.data(), _space_ofs.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        ConvolutionDepthWise_info& operator=(const ConvolutionDepthWise_info&) = delete;


        int num_output;
        int kernel_w;
        int kernel_h;
        int dilation_w;
        int dilation_h;
        int stride_w;
        int stride_h;
        int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
        int pad_right;
        int pad_top;
        int pad_bottom;
        float pad_value;
        int bias_term;

        int weight_data_size;
        int group;

        int int8_scale_term;

        int activation_type;
        bool use_int8_requantize;

        const CudaMat* const gpu_activation_params;
        const CudaMat* const gpu_weight_data;
        const CudaMat* const gpu_bias_data;
        const CudaMat* const gpu_weight_data_int8_scales;
        const CudaMat* const gpu_bottom_blob_int8_scales;

        const float* const gpu_top_blob_int8_scale;

        std::shared_ptr<int> gpu_space_ofs;

        int kernel_extent_w;
        int kernel_extent_h;
        int maxk;

    };

    ConvolutionDepthWise_cuda();
    ~ConvolutionDepthWise_cuda();

    virtual int load_param(const ParamDict& pd);

    using ConvolutionDepthWise::load_model;
    virtual int load_model(const CudaModelBinFromMatArray& mb);
    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);

    using ConvolutionDepthWise::forward;
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

protected:
    void make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const;

    int forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

    std::shared_ptr<ncnn::CudaAllocator> _cuda_allocator;

public:
    CudaMat gpu_activation_params;

    // model
    CudaMat gpu_weight_data;
    CudaMat gpu_bias_data;

    CudaMat gpu_weight_data_int8_scales;
    CudaMat gpu_bottom_blob_int8_scales;
    float* gpu_top_blob_int8_scale;


};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_CUDA_H
