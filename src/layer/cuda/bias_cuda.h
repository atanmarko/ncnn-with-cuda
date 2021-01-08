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


#ifndef LAYER_BIAS_CUDA_H
#define LAYER_BIAS_CUDA_H

#include "bias.h"

namespace ncnn {

class Bias_cuda : virtual public Bias
{
public:
	Bias_cuda();

    virtual int load_model(const CudaModelBinFromMatArray& mb);
    virtual int load_model(const  ModelBin& mb);

    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;

protected:

    // model
    CudaMat gpu_bias_data;

    std::shared_ptr<ncnn::CudaAllocator> _cuda_allocator;
};

} // namespace ncnn

#endif // LAYER_BIAS_CUDA_H
