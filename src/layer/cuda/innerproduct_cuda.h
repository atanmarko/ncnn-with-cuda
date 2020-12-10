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


#ifndef LAYER_INNERPRODUCT_CUDA_H
#define LAYER_INNERPRODUCT_CUDA_H

#include "innerproduct.h"

namespace ncnn {

class InnerProduct_cuda : virtual public InnerProduct
{
public:
    struct InnerProduct_info
    {
        InnerProduct_info(const InnerProduct_cuda& info)
            : num_output(info.num_output), bias_term(info.bias_term), weight_data_size(info.weight_data_size),
              int8_scale_term(info.int8_scale_term), activation_type(info.activation_type),
              gpu_activation_params(&info.gpu_activation_params),
              gpu_weight_data(&info.gpu_weight_data),
              gpu_bias_data(&info.gpu_bias_data),
              gpu_weight_data_int8_scales(&info.gpu_weight_data_int8_scales),
              gpu_bottom_blob_int8_scale(info.gpu_bottom_blob_int8_scale)
        {

        }

        int num_output;
        int bias_term;
        int weight_data_size;
        int int8_scale_term;
        int activation_type;
        const CudaMat* const gpu_activation_params;
        const CudaMat* const gpu_weight_data;
        const CudaMat* const gpu_bias_data;
        const CudaMat* const gpu_weight_data_int8_scales;
        const float* const gpu_bottom_blob_int8_scale;

    };

    InnerProduct_cuda();
    ~InnerProduct_cuda();


    virtual int load_param(const ParamDict& pd);

    using InnerProduct::load_model;
    virtual int load_model(const CudaModelBinFromMatArray& mb);

    virtual int create_pipeline(const Option& opt);

    using InnerProduct::forward;
    using InnerProduct::forward_int8;
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;
    virtual int forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

public:
    CudaMat gpu_activation_params;

    // model
    CudaMat gpu_weight_data;
    CudaMat gpu_bias_data;

    CudaMat gpu_weight_data_int8_scales;
    float* gpu_bottom_blob_int8_scale;

private:
    float* gpu_scratch_pad_memory;
    const int gpu_scratch_pad_memory_size{1024*1024}; //IMPORTANT: matrix size limited by scratchpad memory

};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_CUDA_H
