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


#include "binaryop_cuda.h"

#include <math.h>

namespace ncnn {


int binary_op_cuda_forward(const CudaMat& a, const CudaMat& b, CudaMat& c, const Option& opt, int op_type);
int binary_op_scalar_inplace_cuda_forward(CudaMat& a, float b, const Option& opt, int op_type);

BinaryOp_cuda::BinaryOp_cuda()
{
    support_cuda = true;
}

int BinaryOp_cuda::create_pipeline(const Option& opt)
{
    BinaryOp::create_pipeline(opt);

    return 0;
}

int BinaryOp_cuda::destroy_pipeline(const Option& opt)
{
    BinaryOp::destroy_pipeline(opt);
    return 0;
}

int BinaryOp_cuda::load_param(const ParamDict& pd)
{
    BinaryOp::load_param(pd);

    return 0;
}



// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting


int BinaryOp_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const
{
    const CudaMat& bottom_blob = bottom_blobs[0];
    const CudaMat& bottom_blob1 = bottom_blobs[1];

    CudaMat& top_blob = top_blobs[0];

    return binary_op_cuda_forward(bottom_blob, bottom_blob1, top_blob, opt, this->op_type);
}

int BinaryOp_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const
{
    return binary_op_scalar_inplace_cuda_forward(bottom_top_blob, b, opt, this->op_type);
}

} // namespace ncnn
