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


#ifndef LAYER_CONVOLUTION_CUDA_H
#define LAYER_CONVOLUTION_CUDA_H

#include "convolution.h"

namespace ncnn {

class Convolution_cuda : virtual public Convolution
{
public:
    Convolution_cuda();

    virtual int load_param(const ParamDict& pd);

    using Convolution::load_model;
    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);

    using Convolution::forward;
    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

protected:
    void make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const;

public:
    CudaMat gpu_activation_params;

    // model
    CudaMat gpu_weight_data;
    CudaMat gpu_bias_data;

    CudaMat gpu_weight_data_int8_scales;
    float* gpu_bottom_blob_int8_scale;

};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_CUDA_H
