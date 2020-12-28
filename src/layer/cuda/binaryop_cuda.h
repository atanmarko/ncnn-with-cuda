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


#ifndef LAYER_BINARYOP_CUDA_H
#define LAYER_BINARYOP_CUDA_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_cuda : virtual public BinaryOp
{
public:
    BinaryOp_cuda();

    virtual int create_pipeline(const Option& /*opt*/);
    virtual int destroy_pipeline(const Option& /*opt*/);

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;
    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;

};

} // namespace ncnn

#endif // LAYER_BINARYOP_CUDA_H
