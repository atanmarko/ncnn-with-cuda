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


#ifndef LAYER_PADDING_CUDA_H
#define LAYER_PADDING_CUDA_H

#include "padding.h"

namespace ncnn {

enum class PaddingVariableType
{
    type_char = 0,
    type_unsigned_short  = 1,
    type_float = 2
};

union PaddingValue {
    char c;
    unsigned short sh;
    float fl;
};

template<typename T>
struct GPUPaddingValue {
    T* per_channel_values;
    T value;
    int per_channel_pad_data_size{0};
};

class Padding_cuda : virtual public Padding
{
public:
    Padding_cuda();

    ~Padding_cuda();

    virtual int load_param(const ParamDict& pd);

    using Padding::load_model;

    virtual int load_model(const CudaModelBinFromMatArray& pd);
    virtual int load_model(const ModelBin& pd);

    using Padding::forward;

    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;

    CudaMat gpu_per_channel_pad_data;
    char* gpu_per_channel_pad_data_char{nullptr};
    unsigned short* gpu_per_channel_pad_data_unsigned_short{nullptr};
    unsigned short* gpu_per_channel_pad_data_unsigned_short_use_fp16{nullptr};
    float* gpu_per_channel_pad_data_float{nullptr};

};

} // namespace ncnn

#endif // LAYER_PADDING_CUDA_H
