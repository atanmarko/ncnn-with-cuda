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


#ifndef LAYER_SOFTMAX_CUDA_H
#define LAYER_SOFTMAX_CUDA_H

#include "softmax.h"

namespace ncnn {

class Softmax_cuda : virtual public Softmax
{
public:
    Softmax_cuda();

    ~Softmax_cuda();


    virtual int load_param(const ParamDict& pd);

    using Softmax::forward_inplace;

    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;

private:
    float* gpu_scratch_pad_memory;
    const int gpu_scratch_pad_memory_size{10*1024*1024}; //IMPORTANT: matrix size limited by scratchpad memory

};

} // namespace ncnn

#endif // LAYER_SOFTMAX_CUDA_H
